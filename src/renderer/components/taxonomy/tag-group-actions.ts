import type { QueryClient } from '@tanstack/react-query';
import { invoke } from '../../platform';
import type { TagGrouping } from './tag-groups';

type CachedCategory = { label: string; tagGrouping?: TagGrouping };
type ToastSender = { send: (event: any) => void };

// Flip a category's tag-grouping toggle. Optimistic — the panel re-lays out
// immediately — and rolled back with a toast if the write fails. Shared by
// the sidebar's quick toggle and the edit-category modal.
export async function setCategoryTagGrouping(
  queryClient: QueryClient,
  libraryService: ToastSender,
  categoryLabel: string,
  mode: TagGrouping
) {
  const snapshot = queryClient.getQueriesData<CachedCategory[]>({
    queryKey: ['taxonomy', 'categories'],
  });
  queryClient.setQueriesData<CachedCategory[]>(
    { queryKey: ['taxonomy', 'categories'] },
    (old) =>
      old?.map((c) =>
        c.label === categoryLabel ? { ...c, tagGrouping: mode } : c
      )
  );
  try {
    await invoke('update-category-tag-grouping', [categoryLabel, mode]);
    queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
  } catch (e) {
    for (const [key, value] of snapshot) {
      queryClient.setQueryData(key, value);
    }
    libraryService.send({
      type: 'ADD_TOAST',
      data: {
        type: 'error',
        title: 'Failed to update tag grouping',
        message: String(e),
      },
    });
  }
}
