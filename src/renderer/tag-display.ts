// Shared display rules for tags.
//
// A face cluster whose name collides with a hand-curated tag is stored as
// "<name>_cluster" (tag labels are globally unique server-side; see
// media/people.go PersonClusterSuffix). The suffix is an implementation
// detail: every UI surface shows the plain name and keeps the real label in
// values/queries.
export const CLUSTER_SUFFIX = '_cluster';

export function displayTagLabel(label: string): string {
  return label.endsWith(CLUSTER_SUFFIX)
    ? label.slice(0, -CLUSTER_SUFFIX.length)
    : label;
}

