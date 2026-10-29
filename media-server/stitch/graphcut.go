package stitch

// maxflow is Dinic's algorithm on an adjacency-list graph with int64
// capacities: small and dependable for the grid graphs a seam is cut in
// (a band a few dozen pixels wide round a mask, tens of thousands of nodes).
type maxflow struct {
	n     int
	head  []int32
	next  []int32
	to    []int32
	cap   []int64
	level []int32
	iter  []int32
}

func newMaxflow(n int) *maxflow {
	g := &maxflow{n: n, head: make([]int32, n)}
	for i := range g.head {
		g.head[i] = -1
	}
	return g
}

// edge adds u→v with capacity c and v→u with capacity rc (each is the
// other's residual).
func (g *maxflow) edge(u, v int, c, rc int64) {
	g.to = append(g.to, int32(v))
	g.cap = append(g.cap, c)
	g.next = append(g.next, g.head[u])
	g.head[u] = int32(len(g.to) - 1)
	g.to = append(g.to, int32(u))
	g.cap = append(g.cap, rc)
	g.next = append(g.next, g.head[v])
	g.head[v] = int32(len(g.to) - 1)
}

func (g *maxflow) bfs(s, t int) bool {
	if g.level == nil {
		g.level = make([]int32, g.n)
	}
	for i := range g.level {
		g.level[i] = -1
	}
	q := make([]int32, 0, g.n)
	g.level[s] = 0
	q = append(q, int32(s))
	for k := 0; k < len(q); k++ {
		u := q[k]
		for e := g.head[u]; e >= 0; e = g.next[e] {
			if g.cap[e] > 0 && g.level[g.to[e]] < 0 {
				g.level[g.to[e]] = g.level[u] + 1
				q = append(q, g.to[e])
			}
		}
	}
	return g.level[t] >= 0
}

// dfs pushes a blocking flow, iteratively (paths can be thousands of nodes long).
func (g *maxflow) dfs(s, t int) int64 {
	var total int64
	type frame struct{ u int32 }
	stack := make([]int32, 0, 64) // the edges on the current path
	u := int32(s)
	for {
		if int(u) == t {
			// bottleneck along the path, then push it
			f := int64(1) << 62
			for _, e := range stack {
				if g.cap[e] < f {
					f = g.cap[e]
				}
			}
			for _, e := range stack {
				g.cap[e] -= f
				g.cap[e^1] += f
			}
			total += f
			// back up to the first saturated edge
			k := 0
			for k < len(stack) && g.cap[stack[k]] > 0 {
				k++
			}
			stack = stack[:k]
			if k == 0 {
				u = int32(s)
			} else {
				u = g.to[stack[k-1]]
			}
			continue
		}
		advanced := false
		for ; g.iter[u] >= 0; g.iter[u] = g.next[g.iter[u]] {
			e := g.iter[u]
			v := g.to[e]
			if g.cap[e] > 0 && g.level[v] == g.level[u]+1 {
				stack = append(stack, e)
				u = v
				advanced = true
				break
			}
		}
		if advanced {
			continue
		}
		// dead end: retreat
		g.level[u] = -1
		if len(stack) == 0 {
			return total
		}
		stack = stack[:len(stack)-1]
		if len(stack) == 0 {
			u = int32(s)
		} else {
			u = g.to[stack[len(stack)-1]]
		}
	}
}

func (g *maxflow) run(s, t int) int64 {
	var flow int64
	g.iter = make([]int32, g.n)
	for g.bfs(s, t) {
		copy(g.iter, g.head)
		flow += g.dfs(s, t)
	}
	return flow
}

// sourceSide marks the nodes still reachable from s once the flow is maximal:
// one side of the minimum cut.
func (g *maxflow) sourceSide(s int) []bool {
	seen := make([]bool, g.n)
	q := []int32{int32(s)}
	seen[s] = true
	for k := 0; k < len(q); k++ {
		u := q[k]
		for e := g.head[u]; e >= 0; e = g.next[e] {
			if g.cap[e] > 0 && !seen[g.to[e]] {
				seen[g.to[e]] = true
				q = append(q, g.to[e])
			}
		}
	}
	return seen
}
