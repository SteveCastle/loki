// Package features holds compile-time feature switches for the media server.
package features

// Dream reports whether this binary exposes loki-dream (Kromoss, a MOSS-Image
// fine-tune): the "dream" task, the scene editor's "new scene from a prompt"
// API, the setup wizard's image-generation group and the loki-dream / kromoss
// downloads. It is off unless the server is built with `-tags dream`
// (`LOKI_DREAM=1 npm run build:server`), so releases ship without
// text-to-image. Image and video editing (retouch / reshoot) are unaffected.
const Dream = dreamBuild
