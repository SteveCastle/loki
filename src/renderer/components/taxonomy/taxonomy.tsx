import { useState, useContext, useRef, useEffect, useMemo } from 'react';
import { useSelector } from '@xstate/react';
import { useQuery } from '@tanstack/react-query';
import { debounce } from 'lodash';
import Fuse from 'fuse.js';
import { Tooltip } from 'react-tooltip';
import { GlobalStateContext } from '../../state';
import { FilterModeOption, getNextFilterMode } from '../../../settings';
import restart from '../../../../assets/restart.svg';
import union from '../../../../assets/union.svg';
import clear from '../../../../assets/cancel.svg';
import intersect from '../../../../assets/intersect.svg';
import selective from '../../../../assets/selective.svg';
import group from '../../../../assets/group.svg';
import addMediaImage from '../../../../assets/add-media-image.svg';
import Skeleton, { SkeletonTheme } from 'react-loading-skeleton';
import 'react-loading-skeleton/dist/skeleton.css';

import Tag from './tag';
import VirtualizedTagGrid from './virtualized-tag-grid';
import TagListView from './tag-list-view';
import NewTagModal from './new-tag-modal';
import NewCategoryModal from './new-category-modal';
import './taxonomy.css';
import Category from './category';
import { invoke } from '../../platform';
import QueryInput from '../query-input/QueryInput';

const VIRTUALIZE_THRESHOLD = 300;
// Hard cap on search results. Each rendered <Tag> fires an IPC call to
// fetch its preview, so an unbounded match list (e.g. 1-char query against
// a 50k-tag library) used to flood the renderer + main process and freeze
// the app. Fuse still ranks across the full set; we only render the top N.
const MAX_SEARCH_RESULTS = 200;

type Concept = {
  label: string;
  category: string;
  weight: number;
  description: string;
};

type TagViewMode = 'card' | 'list';

type Category = {
  label: string;
  weight: number;
  description: string;
  tagViewMode?: TagViewMode;
};

type FilterModeIconMap = {
  [key in FilterModeOption]: string;
};

// Loaders for the three taxonomy slices. Each is a separate React Query so
// the panel can fetch categories cheaply, then lazy-load per-category tags
// and the full tag list (for search) only when needed.
async function loadCategories(): Promise<Category[]> {
  const result = await invoke('load-categories', []);
  return (result as Category[]) ?? [];
}

async function loadCategoryTags(categoryLabel: string): Promise<Concept[]> {
  if (!categoryLabel) return [];
  const result = await invoke('load-category-tags', [categoryLabel]);
  return (result as Concept[]) ?? [];
}

async function loadAllTags(): Promise<Concept[]> {
  const result = await invoke('load-all-tags', []);
  return (result as Concept[]) ?? [];
}

const filteringModeIcons: FilterModeIconMap = {
  OR: union,
  AND: intersect,
  EXCLUSIVE: selective,
};

export default function Taxonomy() {
  const categoryListRef = useRef<HTMLDivElement>(null);
  const { libraryService } = useContext(GlobalStateContext);
  const selectedTags = useSelector(libraryService, (state) => {
    return state.context.dbQuery.tags;
  });
  const initSessionId = useSelector(
    libraryService,
    (state) => state.context.initSessionId
  );
  const { applyTagPreview, filteringMode, applyTagToAll, libraryLayout } =
    useSelector(libraryService, (state) => {
      return state.context.settings;
    });

  const textFilter = useSelector(
    libraryService,
    (state) => state.context.textFilter
  );

  const state = useSelector(
    libraryService,
    (state) => state,
    (a, b) => {
      return a.matches(b);
    }
  );
  const [newTextFilter, setNewTextFilter] = useState<string>('');

  function setTextFilter(text: string) {
    libraryService.send({
      type: 'SET_TEXT_FILTER',
      data: { textFilter: text },
    });
  }

  // If textFilter changes, update the input field
  useEffect(() => {
    if (textFilter !== newTextFilter) {
      setNewTextFilter(textFilter);
    }
  }, [textFilter]);

  // Two-tier search state: tagFilterInput updates on every keystroke so the
  // input feels responsive; tagFilter is the debounced value dispatched to the
  // search worker. The actual fuzzy match runs off-thread (see the worker
  // setup below), so the debounce only needs to coalesce bursts of keystrokes
  // rather than guard the main thread — hence a short 150ms window.
  const [tagFilterInput, setTagFilterInput] = useState<string>('');
  const [tagFilter, setTagFilter] = useState<string>('');
  // Set once the search input has been focused. Used to warm the full-tag
  // fetch (and worker index) before the first keystroke so the initial search
  // isn't stalled waiting on the network. Stays true for the session — with
  // staleTime: Infinity the data is fetched at most once.
  const [searchFocused, setSearchFocused] = useState<boolean>(false);
  const debouncedSetTagFilter = useRef(
    debounce((value: string) => {
      setTagFilter(value);
    }, 150)
  );
  useEffect(() => {
    debouncedSetTagFilter.current(tagFilterInput);
  }, [tagFilterInput]);
  useEffect(() => {
    const debounced = debouncedSetTagFilter.current;
    return () => {
      debounced.cancel();
    };
  }, []);

  const [addingTag, setAddingTag] = useState<boolean>(false);
  const [addingCategory, setAddingCategory] = useState<boolean>(false);
  const [editingTag, setEditingTag] = useState<string | null>(null);
  const [editingCategory, setEditingCategory] = useState<string | null>(null);
  const activeCategory = useSelector(
    libraryService,
    (state) => state.context.activeCategory
  );

  const isDisabled =
    state.matches({ library: 'boot' }) ||
    state.matches({ library: 'selectingDB' }) ||
    state.matches({ library: 'loadingFromFS' }) ||
    state.matches({ library: 'loadingDB' });

  function setActiveCategory(category: string) {
    libraryService.send({
      type: 'SET_ACTIVE_CATEGORY',
      data: { category },
    });
  }

  // User-initiated category click: clears any active search before switching
  // categories so the picked category is what the user actually sees.
  function handleCategoryClick(category: string) {
    if (tagFilterInput || tagFilter) {
      setTagFilterInput('');
      setTagFilter('');
      debouncedSetTagFilter.current.cancel();
    }
    setActiveCategory(category);
  }
  // Three separate queries, all under the 'taxonomy' prefix so existing
  // broad invalidations (`queryClient.invalidateQueries(['taxonomy'])`) keep
  // working. staleTime: Infinity means remounts within a session never refetch
  // — mutations and DB swaps (initSessionId) are what trigger reloads.
  const { data: categories } = useQuery<Category[], Error>(
    ['taxonomy', 'categories', initSessionId],
    loadCategories,
    { staleTime: Infinity }
  );

  const { data: activeCategoryTags, isFetching: isFetchingCategoryTags } =
    useQuery<Concept[], Error>(
      ['taxonomy', 'category-tags', activeCategory, initSessionId],
      () => loadCategoryTags(activeCategory),
      { enabled: !!activeCategory, staleTime: Infinity }
    );

  // Warmed as soon as the search input gains focus (or, as a fallback, on the
  // first keystroke) so the fetch + worker indexing overlap with the user
  // composing their query — by the time they type, the data is usually already
  // cached and indexed, so the initial search isn't stalled on the network.
  const { data: allTagsData, isFetching: isFetchingAllTags } = useQuery<
    Concept[],
    Error
  >(['taxonomy', 'all-tags', initSessionId], loadAllTags, {
    enabled: searchFocused || !!tagFilterInput,
    staleTime: Infinity,
  });

  // Display categories alphabetically regardless of their stored weight order.
  // Weight still controls drag-reorder persistence and edit-modal logic; we
  // only sort here for the rendered list, the auto-select fallback, and the
  // scroll-to-active behaviour below.
  const sortedCategories = useMemo(() => {
    if (!categories) return [] as Category[];
    return [...categories].sort((a, b) =>
      a.label.localeCompare(b.label, undefined, { sensitivity: 'base' })
    );
  }, [categories]);

  // Indexed lookup so the edit-category modal can pull description / view mode
  // without scanning the list every render.
  const categoriesByLabel = useMemo(() => {
    if (!categories) return {} as Record<string, Category>;
    const map: Record<string, Category> = {};
    for (const c of categories) map[c.label] = c;
    return map;
  }, [categories]);

  // Ensure activeCategory is valid; if not, reset to the first available category.
  // Suspended while a search is active — during search no category should be selected.
  useEffect(() => {
    if (!sortedCategories.length) return;
    if (tagFilter) return;
    const exists = sortedCategories.some((c) => c.label === activeCategory);
    if (!exists) {
      setActiveCategory(sortedCategories[0].label);
    }
  }, [sortedCategories, activeCategory, tagFilter]);

  // When a search becomes active, clear the active category — the user is now
  // searching across categories, not viewing one.
  useEffect(() => {
    if (tagFilter) {
      setActiveCategory('');
    }
  }, [tagFilter]);

  // Given every Category is 20 pixels tall, this will scroll to the active category
  // when it is not visible in the list by setting the scrollTop of the categoryListRef

  if (categoryListRef.current && activeCategory && sortedCategories.length) {
    const activeCategoryIndex = sortedCategories.findIndex(
      (category) => category.label === activeCategory
    );
    if (activeCategoryIndex > -1) {
      const activeCategoryTop =
        activeCategoryIndex * 25.5 - categoryListRef.current.clientHeight / 2;
      categoryListRef.current.scrollTop = activeCategoryTop;
    }
  }

  // The full-tag list is only fetched lazily for search; defensive filter
  // drops tags without a label — Fuse and the row components both assume
  // a non-empty string and would otherwise throw when one slips in.
  const allTags = useMemo(() => {
    if (!allTagsData) return [] as Concept[];
    return allTagsData.filter(
      (t) => t && typeof t.label === 'string' && t.label.length > 0
    );
  }, [allTagsData]);

  // Async tag search. Indexing and fuzzy matching run in a Web Worker so a
  // large library (tens of thousands of tags) never blocks the input. Each
  // request carries an id (searchSeq); responses that aren't the latest are
  // dropped so a slow earlier search can't clobber newer results.
  const workerRef = useRef<Worker | null>(null);
  const searchSeq = useRef(0);
  const [searchResults, setSearchResults] = useState<Concept[]>([]);
  // Optimistically assume worker support; flip to false if construction fails
  // (e.g. jsdom in tests) so the synchronous fallback below takes over.
  const [workerReady, setWorkerReady] = useState<boolean>(
    typeof Worker !== 'undefined'
  );

  useEffect(() => {
    if (typeof Worker === 'undefined') {
      setWorkerReady(false);
      return undefined;
    }
    let worker: Worker;
    try {
      worker = new Worker(new URL('./tag-search.worker.ts', import.meta.url));
    } catch {
      setWorkerReady(false);
      return undefined;
    }
    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg?.type === 'result' && msg.id === searchSeq.current) {
        setSearchResults(msg.items as Concept[]);
      }
    };
    workerRef.current = worker;
    setWorkerReady(true);
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  // Keep the worker's index in sync with the loaded tag set. The worker
  // re-runs the outstanding query after re-indexing, so results refresh
  // automatically once data first arrives or after a tag mutation.
  useEffect(() => {
    workerRef.current?.postMessage({ type: 'index', tags: allTags });
  }, [allTags]);

  // Synchronous Fuse fallback — only built/used when no worker is available.
  // Keep these options in sync with tag-search.worker.ts.
  const fallbackFuse = useMemo(() => {
    if (workerReady) return null;
    return new Fuse(allTags, {
      keys: [
        { name: 'label', weight: 2 },
        { name: 'category', weight: 1 },
      ],
      threshold: 0.4,
      ignoreLocation: true,
      minMatchCharLength: 1,
    });
  }, [allTags, workerReady]);

  // Dispatch the debounced query. With a worker the match happens off-thread
  // and arrives via onmessage; without one we fall back to a synchronous
  // search. Bumping searchSeq invalidates any in-flight worker response.
  useEffect(() => {
    const id = (searchSeq.current += 1);
    if (!tagFilter) {
      setSearchResults([]);
    }
    if (workerRef.current) {
      workerRef.current.postMessage({
        type: 'search',
        id,
        query: tagFilter,
        limit: MAX_SEARCH_RESULTS,
      });
    } else if (fallbackFuse && tagFilter) {
      setSearchResults(
        fallbackFuse
          .search(tagFilter, { limit: MAX_SEARCH_RESULTS })
          .map((r) => r.item)
      );
    }
  }, [tagFilter, fallbackFuse]);

  const tags = useMemo(() => {
    if (tagFilter) {
      // Worker-ranked, pre-capped matches across all categories. Briefly empty
      // (or showing the prior query's results) until the worker responds —
      // acceptable for an async search and imperceptible at this debounce.
      return searchResults;
    }
    return (activeCategoryTags ?? [])
      .slice()
      .sort((a, b) => a.weight - b.weight);
  }, [tagFilter, searchResults, activeCategoryTags]);

  if (!categories || state.matches('loadingDB') || state.matches('selectingDB')) {
    return (
      <div className={`Placeholder`}>
        <div className={`inner`}>
          <SkeletonTheme baseColor="#202020" highlightColor="#444">
            <Skeleton />
          </SkeletonTheme>
        </div>
      </div>
    );
  }
  return (
    <>
      <div
        className={`Taxonomy`}
        style={{
          marginTop: libraryLayout === 'left' ? '20px' : '0',
        }}
      >
        <div className="search">
          <QueryInput
            value={newTextFilter}
            onChange={setNewTextFilter}
            onSubmit={(text) => {
              setNewTextFilter(text);
              setTextFilter(text);
            }}
            onClear={() => {
              setNewTextFilter('');
              setTextFilter('');
              setEditingTag(null);
            }}
            disabled={isDisabled}
          />
          <div className="textSearch">
            <input
              type="text"
              placeholder="Search Tags"
              value={tagFilterInput}
              onFocus={() => setSearchFocused(true)}
              onKeyDown={(e) => {
                e.stopPropagation();
              }}
              onKeyUp={(e) => {
                e.stopPropagation();
              }}
              onChange={(e) => setTagFilterInput(e.currentTarget.value)}
            />
            <button
              className="clear-search"
              onClick={() => {
                setTagFilterInput('');
                setTagFilter('');
                debouncedSetTagFilter.current.cancel();
                setEditingTag(null);
              }}
            >
              <img src={clear} />
            </button>
          </div>
        </div>
        <div className="controls">
          <button
            data-tooltip-id="reset-query-tag"
            data-tooltip-delay-show={500}
            onClick={() =>
              libraryService.send({
                type: 'CLEAR_QUERY_TAG',
              })
            }
          >
            <img src={restart} />
          </button>
          <button
            className={'active'}
            data-tooltip-id="filtering-mode"
            data-tooltip-delay-show={500}
            onClick={() =>
              libraryService.send({
                type: 'CHANGE_SETTING',
                data: { filteringMode: getNextFilterMode(filteringMode) },
              })
            }
          >
            <img src={filteringModeIcons[filteringMode]} />
          </button>
          <button
            className={`${applyTagToAll ? 'active' : ''}`}
            data-tooltip-id="apply-tag-to-all"
            data-tooltip-delay-show={500}
            onClick={() =>
              libraryService.send({
                type: 'CHANGE_SETTING',
                data: { applyTagToAll: !applyTagToAll },
              })
            }
          >
            <img src={group} />
          </button>
          <button
            className={`${applyTagPreview ? 'active' : ''}`}
            data-tooltip-id="apply-tag-preview"
            data-tooltip-delay-show={500}
            onClick={() =>
              libraryService.send({
                type: 'CHANGE_SETTING',
                data: { applyTagPreview: !applyTagPreview },
              })
            }
          >
            <img src={addMediaImage} />
          </button>
        </div>
        <div className="categories-column">
          <div
            className={`new-category`}
            onClick={() => setAddingCategory(true)}
          >
            <div className="category-label">+</div>
          </div>
          <div className={`categories`} ref={categoryListRef}>
            {sortedCategories.map((category) => {
              return (
                <Category
                  key={category.label}
                  category={category}
                  activeCategory={activeCategory}
                  setActiveCategory={handleCategoryClick}
                  handleEditAction={setEditingCategory}
                />
              );
            })}
          </div>
        </div>
        {activeCategory && (
          <div className={`new-tag`} onClick={() => setAddingTag(true)}>
            <div className="tag-label">+</div>
          </div>
        )}
        {(() => {
          if (!(activeCategory || tagFilter)) {
            return <div className={`tags`} />;
          }
          // Search results span categories — always use card style.
          // For an active category, honour its persisted tagViewMode.
          const activeViewMode: TagViewMode = tagFilter
            ? 'card'
            : (categoriesByLabel[activeCategory]?.tagViewMode as TagViewMode) ||
              'card';
          // Reordering by drag is meaningless when results are sorted by
          // search relevance — disable DnD while a search is active.
          const disableReorder = !!tagFilter;

          // Skeleton placeholder while the per-category tags (or the full
          // tag list for search) load for the first time. `data === undefined`
          // means React Query hasn't returned yet; once a category is cached
          // or after an optimistic mutation we keep prior data and skip this
          // branch so the panel doesn't flash empty on refetch.
          const isLoadingTags = tagFilter
            ? allTagsData === undefined && isFetchingAllTags
            : activeCategoryTags === undefined && isFetchingCategoryTags;
          if (isLoadingTags) {
            if (activeViewMode === 'list') {
              return (
                <div className="tag-list-view">
                  <SkeletonTheme baseColor="#202020" highlightColor="#444">
                    {Array.from({ length: 16 }).map((_, i) => (
                      <Skeleton
                        key={i}
                        height={28}
                        style={{ marginBottom: 4 }}
                      />
                    ))}
                  </SkeletonTheme>
                </div>
              );
            }
            return (
              <div className="tags">
                <SkeletonTheme baseColor="#202020" highlightColor="#444">
                  {Array.from({ length: 18 }).map((_, i) => (
                    <Skeleton key={i} height={60} />
                  ))}
                </SkeletonTheme>
              </div>
            );
          }
          if (activeViewMode === 'list') {
            return (
              <TagListView
                tags={tags}
                selectedTags={selectedTags}
                isDisabled={isDisabled}
                handleEditAction={setEditingTag}
                disableReorder={disableReorder}
              />
            );
          }
          // Blended search view: when there are matches in the Suggested
          // category, render non-Suggested results as cards on top and
          // Suggested results as a compact list below. Both sections are
          // virtualized — `<Tag>` triggers an IPC preview fetch on mount,
          // so rendering all matches at once would saturate the IPC layer.
          if (tagFilter) {
            const suggestedTags = tags.filter(
              (t: Concept) => t.category === 'Suggested'
            );
            if (suggestedTags.length > 0) {
              const mainTags = tags.filter(
                (t: Concept) => t.category !== 'Suggested'
              );
              return (
                <div className="tags-blended">
                  {mainTags.length > 0 && (
                    <div className="tags-blended-cards">
                      <VirtualizedTagGrid
                        tags={mainTags}
                        selectedTags={selectedTags}
                        isDisabled={isDisabled}
                        handleEditAction={setEditingTag}
                        disableReorder={disableReorder}
                      />
                    </div>
                  )}
                  <div className="suggested-section">
                    <div className="suggested-heading">Suggested</div>
                    <TagListView
                      tags={suggestedTags}
                      selectedTags={selectedTags}
                      isDisabled={isDisabled}
                      handleEditAction={setEditingTag}
                      disableReorder={disableReorder}
                    />
                  </div>
                </div>
              );
            }
          }
          // Virtualize whenever searching (results can be large and each
          // <Tag> mounts an IPC preview fetch), or when browsing a
          // category that exceeds the static threshold.
          if (tagFilter || tags.length > VIRTUALIZE_THRESHOLD) {
            return (
              <VirtualizedTagGrid
                tags={tags}
                selectedTags={selectedTags}
                isDisabled={isDisabled}
                handleEditAction={setEditingTag}
                disableReorder={disableReorder}
              />
            );
          }
          return (
            <div className={`tags`}>
              {tags.map((tag: Concept) => (
                <Tag
                  isDisabled={isDisabled}
                  tags={tags}
                  tag={{
                    label: tag.label,
                    weight: tag.weight,
                    category: tag.category,
                  }}
                  active={selectedTags.includes(tag.label)}
                  handleEditAction={setEditingTag}
                  disableReorder={disableReorder}
                  key={tag.label}
                />
              ))}
            </div>
          );
        })()}
      </div>
      {activeCategory && addingTag ? (
        <NewTagModal
          categoryLabel={activeCategory}
          handleClose={() => setAddingTag(false)}
        />
      ) : null}
      {addingCategory ? (
        <NewCategoryModal
          handleClose={() => setAddingCategory(false)}
          setCategory={setActiveCategory}
        />
      ) : null}
      {editingTag ? (() => {
        // The tag being edited may live outside the active category — in
        // search mode `activeCategory` is intentionally cleared, so fall back
        // to allTagsData to recover the tag's real category and description.
        // categoryLabel isn't used by the edit flow itself (only the create
        // flow consumes it for cache keys), but we pass the real category for
        // correctness.
        const editingTagConcept =
          (activeCategoryTags || []).find(
            (t: Concept) => t.label === editingTag
          ) ||
          (allTagsData || []).find(
            (t: Concept) => t.label === editingTag
          ) ||
          null;
        return (
          <NewTagModal
            categoryLabel={
              editingTagConcept?.category || activeCategory || ''
            }
            handleClose={() => setEditingTag(null)}
            currentValue={editingTag}
            currentDescription={editingTagConcept?.description || ''}
          />
        );
      })() : null}
      {editingCategory ? (
        <NewCategoryModal
          handleClose={() => setEditingCategory(null)}
          setCategory={setActiveCategory}
          currentValue={editingCategory}
          currentDescription={
            categoriesByLabel[editingCategory]?.description || ''
          }
          currentTagViewMode={
            (categoriesByLabel[editingCategory]?.tagViewMode as TagViewMode) ||
            'card'
          }
        />
      ) : null}
      <Tooltip
        id="reset-query-tag"
        content={`Reset tag filter.`}
        place="right"
      />
      <Tooltip
        id="filtering-mode"
        content={`Filtering mode. (Exclusive, Intersection, Union)`}
        place="right"
      />
      <Tooltip
        id="apply-tag-to-all"
        content={`Apply tag to all media in list view.`}
        place="right"
      />
      <Tooltip
        id="apply-tag-preview"
        content={`Update tag image when aplying tag.`}
        place="right"
      />
    </>
  );
}
