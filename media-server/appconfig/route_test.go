package appconfig

import "testing"

func TestParseRoute(t *testing.T) {
	cases := []struct {
		in, want string
		local    bool
		home     bool
	}{
		{"", "local", true, false},
		{"local", "local", true, false},
		{"pool", "pool", false, true},
		{"hybrid", "hybrid", true, true},
		{"Local+Pool", "hybrid", true, true},
		{"pool:Home", "pool:Home", false, true},
		{"pool:Home, cloud,home", "pool:Home,cloud", false, true},
		{"local+pool:Home", "local+pool:Home", true, true},
		{"local+pool:cloud", "local+pool:cloud", true, false},
		{"pool:", "local", true, false},
		{"bogus", "local", true, false},
	}
	for _, c := range cases {
		r := ParseRoute(c.in)
		if got := r.String(); got != c.want {
			t.Errorf("ParseRoute(%q).String() = %q; want %q", c.in, got, c.want)
		}
		if r.Local != c.local || r.UsesPool("home") != c.home {
			t.Errorf("ParseRoute(%q) = %+v; want local=%v usesHome=%v", c.in, r, c.local, c.home)
		}
	}
}
