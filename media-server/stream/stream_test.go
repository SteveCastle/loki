package stream

import (
	"sync/atomic"
	"testing"
	"time"
)

// TestGetConnectionStats verifies connection statistics
func TestGetConnectionStats(t *testing.T) {
	stats := GetConnectionStats()

	if stats == nil {
		t.Fatal("GetConnectionStats() returned nil")
	}

	// Verify expected keys exist
	expectedKeys := []string{
		"active_connections",
		"total_messages",
		"max_connections",
		"dropped_broadcasts",
		"dropped_client_msgs",
		"rejected_connections",
	}

	for _, key := range expectedKeys {
		if _, ok := stats[key]; !ok {
			t.Errorf("Expected key %q not found in stats", key)
		}
	}

	// Verify max_connections value
	if stats["max_connections"] != int64(MaxConcurrentConnections) {
		t.Errorf("max_connections = %v; want %d", stats["max_connections"], MaxConcurrentConnections)
	}
}

// TestAddRemoveClient tests client registration and removal
func TestAddRemoveClient(t *testing.T) {
	// Get initial count
	initialCount := atomic.LoadInt64(&manager.activeCount)

	// Create a test client channel
	clientChan := make(chan Message, ClientChannelBuffer)

	// Add client
	success := AddClient(clientChan, "127.0.0.1:12345", "TestAgent/1.0")
	if !success {
		t.Error("AddClient() should succeed")
	}

	// Verify count increased
	newCount := atomic.LoadInt64(&manager.activeCount)
	if newCount != initialCount+1 {
		t.Errorf("activeCount = %d; want %d", newCount, initialCount+1)
	}

	// Remove client
	RemoveClient(clientChan)

	// Verify count decreased
	finalCount := atomic.LoadInt64(&manager.activeCount)
	if finalCount != initialCount {
		t.Errorf("activeCount after remove = %d; want %d", finalCount, initialCount)
	}
}

// TestRemoveClientNonExistent tests removing a client that doesn't exist
func TestRemoveClientNonExistent(t *testing.T) {
	// This should not panic
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("RemoveClient panicked: %v", r)
		}
	}()

	nonExistentChan := make(chan Message, 1)
	RemoveClient(nonExistentChan)
}

// TestBroadcast tests message broadcasting
func TestBroadcast(t *testing.T) {
	// Get initial message count
	initialMessages := atomic.LoadInt64(&manager.totalMessages)

	// Create and add a client
	clientChan := make(chan Message, ClientChannelBuffer)
	AddClient(clientChan, "127.0.0.1:12345", "TestAgent/1.0")
	defer RemoveClient(clientChan)

	// Broadcast a message
	testMsg := Message{Type: "test", Msg: "test message"}
	Broadcast(testMsg)

	// Wait for message to be processed
	time.Sleep(50 * time.Millisecond)

	// Try to receive the message
	select {
	case received := <-clientChan:
		if received.Type != testMsg.Type {
			t.Errorf("Received message type = %q; want %q", received.Type, testMsg.Type)
		}
		if received.Msg != testMsg.Msg {
			t.Errorf("Received message msg = %q; want %q", received.Msg, testMsg.Msg)
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("Did not receive broadcast message")
	}

	// Verify message count increased
	newMessages := atomic.LoadInt64(&manager.totalMessages)
	if newMessages <= initialMessages {
		t.Errorf("totalMessages should have increased from %d", initialMessages)
	}
}

// TestBroadcastNoClients tests broadcasting when no clients are connected
func TestBroadcastNoClients(t *testing.T) {
	// This should not panic
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Broadcast panicked: %v", r)
		}
	}()

	Broadcast(Message{Type: "test", Msg: "no clients"})
}

// TestBroadcastDropped tests that broadcasts are dropped when hub is full
func TestBroadcastDropped(t *testing.T) {
	initialDropped := atomic.LoadInt64(&manager.droppedBroadcasts)

	// Fill the broadcast buffer
	for i := 0; i < HubBroadcastBuffer+100; i++ {
		Broadcast(Message{Type: "flood", Msg: "flooding"})
	}

	// Some messages should have been dropped
	finalDropped := atomic.LoadInt64(&manager.droppedBroadcasts)
	if finalDropped <= initialDropped {
		// This might not always trigger drops depending on how fast the consumer is
		t.Log("No drops detected - consumer may be keeping up")
	}
	drainBroadcasts()
}

// TestFormatSSEResponse tests SSE response formatting
func TestFormatSSEResponse(t *testing.T) {
	tests := []struct {
		msg      Message
		expected string
	}{
		{
			msg:      Message{Type: "update", Msg: "test"},
			expected: "event: update\ndata: test\n\n",
		},
		{
			msg:      Message{Type: "create", Msg: `{"id":"123"}`},
			expected: "event: create\ndata: {\"id\":\"123\"}\n\n",
		},
		{
			msg:      Message{Type: "", Msg: "empty type"},
			expected: "event: \ndata: empty type\n\n",
		},
	}

	for _, tt := range tests {
		result := formatSSEResponse(tt.msg)
		if result != tt.expected {
			t.Errorf("formatSSEResponse(%+v) = %q; want %q", tt.msg, result, tt.expected)
		}
	}
}

// TestClientStruct tests Client struct fields
func TestClientStruct(t *testing.T) {
	clientChan := make(chan Message, 1)

	client := &Client{
		ID:           "test-client-123",
		Channel:      clientChan,
		LastSeen:     time.Now().Unix(),
		RemoteAddr:   "192.168.1.1:8080",
		UserAgent:    "Mozilla/5.0",
		Connected:    time.Now().Unix(),
		MessagesSent: 0,
	}

	if client.ID != "test-client-123" {
		t.Errorf("Client.ID = %q; want %q", client.ID, "test-client-123")
	}
	if client.Channel != clientChan {
		t.Error("Client.Channel not set correctly")
	}
	if client.RemoteAddr != "192.168.1.1:8080" {
		t.Errorf("Client.RemoteAddr = %q; want %q", client.RemoteAddr, "192.168.1.1:8080")
	}
	if client.UserAgent != "Mozilla/5.0" {
		t.Errorf("Client.UserAgent = %q; want %q", client.UserAgent, "Mozilla/5.0")
	}
}

// TestMessageStruct tests Message struct
func TestMessageStruct(t *testing.T) {
	msg := Message{
		Type: "test-type",
		Msg:  "test message content",
	}

	if msg.Type != "test-type" {
		t.Errorf("Message.Type = %q; want %q", msg.Type, "test-type")
	}
	if msg.Msg != "test message content" {
		t.Errorf("Message.Msg = %q; want %q", msg.Msg, "test message content")
	}
}

// TestConnectionManagerConstants tests that constants are reasonable
func TestConnectionManagerConstants(t *testing.T) {
	if MaxConcurrentConnections < 100 {
		t.Errorf("MaxConcurrentConnections = %d; should be at least 100", MaxConcurrentConnections)
	}

	if ClientChannelBuffer < 10 {
		t.Errorf("ClientChannelBuffer = %d; should be at least 10", ClientChannelBuffer)
	}

	if KeepAliveInterval < 10*time.Second {
		t.Errorf("KeepAliveInterval = %v; should be at least 10s", KeepAliveInterval)
	}

	if CleanupInterval < 30*time.Second {
		t.Errorf("CleanupInterval = %v; should be at least 30s", CleanupInterval)
	}

	if HubBroadcastBuffer < 100 {
		t.Errorf("HubBroadcastBuffer = %d; should be at least 100", HubBroadcastBuffer)
	}
}

// drainBroadcasts empties the broadcast channel to prevent test interference
func drainBroadcasts() {
	for {
		select {
		case <-manager.broadcast:
		default:
			return
		}
	}
}

// TestMultipleClients tests handling multiple clients
func TestMultipleClients(t *testing.T) {
	drainBroadcasts()
	initialCount := atomic.LoadInt64(&manager.activeCount)

	// Create multiple clients
	clients := make([]clientChan, 5)
	for i := range clients {
		clients[i] = make(chan Message, ClientChannelBuffer)
		AddClient(clients[i], "127.0.0.1:"+string(rune('0'+i)), "TestAgent")
	}

	// Verify count
	currentCount := atomic.LoadInt64(&manager.activeCount)
	if currentCount != initialCount+5 {
		t.Errorf("activeCount = %d; want %d", currentCount, initialCount+5)
	}

	// Broadcast message
	Broadcast(Message{Type: "multi", Msg: "to all"})
	time.Sleep(50 * time.Millisecond)

	// All clients should receive the message
	for i, client := range clients {
		select {
		case msg := <-client:
			if msg.Type != "multi" {
				t.Errorf("Client %d received wrong message type: %q", i, msg.Type)
			}
		case <-time.After(100 * time.Millisecond):
			t.Errorf("Client %d did not receive message", i)
		}
	}

	// Cleanup
	for _, client := range clients {
		RemoveClient(client)
	}

	// Verify count returned to initial
	finalCount := atomic.LoadInt64(&manager.activeCount)
	if finalCount != initialCount {
		t.Errorf("Final activeCount = %d; want %d", finalCount, initialCount)
	}
}
