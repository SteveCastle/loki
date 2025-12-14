package stream

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// Maximum number of concurrent SSE connections allowed
	MaxConcurrentConnections = 5000
	// Buffer size for each client's message channel
	ClientChannelBuffer = 256
	// How often to send keep-alive messages
	KeepAliveInterval = 30 * time.Second
	// How often to cleanup dead connections
	CleanupInterval = 60 * time.Second
	// Buffer size for hub broadcast queue
	HubBroadcastBuffer = 2048
)

type clientChan chan Message

// Client represents a connected SSE client
type Client struct {
	ID           string
	Channel      clientChan
	LastSeen     int64 // Unix timestamp
	RemoteAddr   string
	UserAgent    string
	Connected    int64 // Unix timestamp when connected
	MessagesSent int64
}

// ConnectionManager manages SSE client connections with production-ready features
type ConnectionManager struct {
	clients           sync.Map // map[clientChan]*Client
	activeCount       int64    // Atomic counter for active connections
	totalMessages     int64    // Atomic counter for total messages sent
	broadcast         chan Message
	droppedBroadcasts int64
	droppedClientMsgs int64
	rejectedConns     int64
	mu                sync.RWMutex
	shutdown          chan struct{}
	shutdownOnce      sync.Once
}

var manager *ConnectionManager

func init() {
	manager = &ConnectionManager{
		shutdown:  make(chan struct{}),
		broadcast: make(chan Message, HubBroadcastBuffer),
	}
	// Start background loops
	go manager.runBroadcastLoop()
	go manager.cleanupRoutine()
}

type Message struct {
	Type string `json:"type"`
	Msg  string `json:"msg"`
}

// GetConnectionStats returns current connection statistics
func GetConnectionStats() map[string]interface{} {
	return map[string]interface{}{
		"active_connections":   atomic.LoadInt64(&manager.activeCount),
		"total_messages":       atomic.LoadInt64(&manager.totalMessages),
		"max_connections":      MaxConcurrentConnections,
		"dropped_broadcasts":   atomic.LoadInt64(&manager.droppedBroadcasts),
		"dropped_client_msgs":  atomic.LoadInt64(&manager.droppedClientMsgs),
		"rejected_connections": atomic.LoadInt64(&manager.rejectedConns),
	}
}

// AddClient registers a new client connection with production safeguards
func AddClient(c clientChan, remoteAddr, userAgent string) bool {
	// Check connection limit
	if atomic.LoadInt64(&manager.activeCount) >= MaxConcurrentConnections {
		atomic.AddInt64(&manager.rejectedConns, 1)
		log.Printf("Connection limit reached (%d), rejecting new client from %s", MaxConcurrentConnections, remoteAddr)
		return false
	}

	client := &Client{
		ID:         fmt.Sprintf("%d-%s", time.Now().UnixNano(), remoteAddr),
		Channel:    c,
		LastSeen:   time.Now().Unix(),
		RemoteAddr: remoteAddr,
		UserAgent:  userAgent,
		Connected:  time.Now().Unix(),
	}

	manager.clients.Store(c, client)
	atomic.AddInt64(&manager.activeCount, 1)

	log.Printf("Client connected: %s (total: %d)", client.ID, atomic.LoadInt64(&manager.activeCount))
	return true
}

// RemoveClient removes a client connection and cleans up resources
func RemoveClient(c clientChan) {
	if client, exists := manager.clients.LoadAndDelete(c); exists {
		clientData := client.(*Client)
		atomic.AddInt64(&manager.activeCount, -1)

		// Safely close the channel
		select {
		case <-c: // Drain any remaining messages
		default:
		}
		close(c)

		log.Printf("Client disconnected: %s (total: %d)", clientData.ID, atomic.LoadInt64(&manager.activeCount))
	}
}

// Broadcast enqueues a message for fan-out without blocking callers
func Broadcast(msg Message) {
	if manager == nil {
		return
	}
	select {
	case manager.broadcast <- msg:
		// ok: dispatcher will fan-out and account metrics
	default:
		// hub busy; drop to protect producers
		atomic.AddInt64(&manager.droppedBroadcasts, 1)
	}
}

// runBroadcastLoop fans out messages to clients without blocking
func (cm *ConnectionManager) runBroadcastLoop() {
	for {
		select {
		case msg := <-cm.broadcast:
			cm.clients.Range(func(key, value any) bool {
				c := key.(clientChan)
				client := value.(*Client)
				select {
				case c <- msg:
					atomic.StoreInt64(&client.LastSeen, time.Now().Unix())
					atomic.AddInt64(&client.MessagesSent, 1)
					atomic.AddInt64(&cm.totalMessages, 1)
				default:
					// client queue full; drop this message for this client
					atomic.AddInt64(&cm.droppedClientMsgs, 1)
				}
				return true
			})
		case <-cm.shutdown:
			return
		}
	}
}

// cleanupRoutine periodically removes stale connections
func (cm *ConnectionManager) cleanupRoutine() {
	ticker := time.NewTicker(CleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			cm.cleanupStaleConnections()
		case <-cm.shutdown:
			return
		}
	}
}

// cleanupStaleConnections removes clients that haven't been seen recently
func (cm *ConnectionManager) cleanupStaleConnections() {
	now := time.Now().Unix()
	staleThreshold := now - int64(CleanupInterval.Seconds()*2) // 2x cleanup interval

	var staleClients []clientChan

	cm.clients.Range(func(key, value any) bool {
		c := key.(clientChan)
		client := value.(*Client)

		if atomic.LoadInt64(&client.LastSeen) < staleThreshold {
			staleClients = append(staleClients, c)
		}
		return true
	})

	if len(staleClients) > 0 {
		log.Printf("Cleaning up %d stale connections", len(staleClients))
		for _, staleClient := range staleClients {
			RemoveClient(staleClient)
		}
	}
}

// Shutdown gracefully shuts down the connection manager
func Shutdown() {
	manager.shutdownOnce.Do(func() {
		close(manager.shutdown)

		// Close all client connections
		manager.clients.Range(func(key, value any) bool {
			c := key.(clientChan)
			RemoveClient(c)
			return true
		})

		log.Println("Stream connection manager shutdown complete")
	})
}

// StreamHandler handles the SSE endpoint with production-ready features
func StreamHandler(w http.ResponseWriter, r *http.Request) {
	// Check if we can accept more connections
	if atomic.LoadInt64(&manager.activeCount) >= MaxConcurrentConnections {
		http.Error(w, "Server at capacity, please try again later", http.StatusServiceUnavailable)
		return
	}

	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Cache-Control")
	w.Header().Del("Content-Encoding")

	// Check if client supports streaming
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}

	// Create buffered channel for this client
	messageChan := make(chan Message, ClientChannelBuffer)
	remoteAddr := r.RemoteAddr
	userAgent := r.UserAgent()

	// Try to add the client
	if !AddClient(messageChan, remoteAddr, userAgent) {
		http.Error(w, "Server at capacity", http.StatusServiceUnavailable)
		return
	}

	// Ensure cleanup on disconnect
	defer RemoveClient(messageChan)

	// Get request context for cancellation detection
	ctx := r.Context()

	// Setup keep-alive ticker
	keepAliveTicker := time.NewTicker(KeepAliveInterval)
	defer keepAliveTicker.Stop()

	// Send initial connection confirmation
	if _, err := io.WriteString(w, "data: {\"type\":\"connected\",\"msg\":\"SSE connection established\"}\n\n"); err != nil {
		return
	}
	flusher.Flush()

	for {
		select {
		case <-ctx.Done():
			// Client disconnected
			return

		case msg := <-messageChan:
			// Send message to client
			if _, err := io.WriteString(w, formatSSEResponse(msg)); err != nil {
				return // Connection broken
			}
			flusher.Flush()

		case <-keepAliveTicker.C:
			// Send keep-alive message
			if _, err := io.WriteString(w, ": keep-alive\n\n"); err != nil {
				return // Connection broken
			}
			flusher.Flush()
		}
	}
}

func formatSSEResponse(msg Message) string {
	return fmt.Sprintf("event: %s\ndata: %s\n\n", msg.Type, msg.Msg)
}
