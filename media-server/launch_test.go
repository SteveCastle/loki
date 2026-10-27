package main

import (
	"os"
	"testing"
)

func TestLaunchedInBackground(t *testing.T) {
	orig := os.Args
	t.Cleanup(func() { os.Args = orig })

	cases := []struct {
		args []string
		env  string
		want bool
	}{
		{args: nil, want: false},
		{args: []string{"--background"}, want: true},
		{args: []string{"-background"}, want: true},
		{args: []string{"/Background"}, want: true},
		{args: []string{"--verbose"}, want: false},
		{env: "1", want: true},
		{env: "true", want: true},
		{env: "0", want: false},
	}
	for _, c := range cases {
		os.Args = append([]string{"lowkeymediaserver"}, c.args...)
		t.Setenv("LOWKEY_NO_BROWSER", c.env)
		if got := launchedInBackground(); got != c.want {
			t.Errorf("args=%v env=%q: got %v, want %v", c.args, c.env, got, c.want)
		}
	}
}
