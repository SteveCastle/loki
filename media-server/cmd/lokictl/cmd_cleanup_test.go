package main

import "testing"

func TestMediaCleanupBuildsJobInput(t *testing.T) {
	cases := []struct {
		args []string
		want string
	}{
		{[]string{"--detach"}, "cleanup"},
		{[]string{"--dry-run", "--detach"}, "cleanup --dry-run"},
		{[]string{"--max-missing-percent", "100", "--skip-orphans", "--detach"}, "cleanup --skip-orphans --max-missing-percent 100"},
		{[]string{"--max-missing-percent=12.5", "--detach"}, "cleanup --max-missing-percent 12.5"},
		{[]string{"--dir", "D:/my pics", "--dry-run", "--detach"}, `cleanup --dry-run --dir "D:/my pics"`},
		{[]string{"D:/pics", "--detach"}, "cleanup --dir D:/pics"},
	}
	for _, c := range cases {
		srv, input := createCapture(t)
		a, _, errOut := appForServer(srv.URL)
		if code := cmdMediaCleanup(a, c.args); code != 0 {
			t.Errorf("%v: exit %d, stderr %s", c.args, code, errOut.String())
			continue
		}
		if *input != c.want {
			t.Errorf("%v: input = %q, want %q", c.args, *input, c.want)
		}
	}
}

func TestMediaCleanupRejectsBadArgs(t *testing.T) {
	for _, args := range [][]string{
		{"--bogus"},
		{"--dir"},
		{"--max-missing-percent", "abc"},
		{"--max-missing-percent", "150"},
		{"D:/a", "D:/b"},
	} {
		srv, _ := createCapture(t)
		a, _, _ := appForServer(srv.URL)
		if code := cmdMediaCleanup(a, args); code != 2 {
			t.Errorf("%v: exit %d, want 2 (usage)", args, code)
		}
	}
}
