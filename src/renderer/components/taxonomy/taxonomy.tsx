import { useState, useContext, useRef, useEffect, useMemo } from 'react';
import { useSelector } from '@xstate/react';
import { useQuery } from '@tanstack/react-query';
import { debounce } from 'lodash';
import { Tooltip } from 'react-tooltip';
import { GlobalStateContext } from '../../state';
import { getNextFilterMode } from '../../../settings';
import restart from '../../../../assets/restart.svg';
import group from '../../../../assets/group.svg';
import addMediaImage from '../../../../assets/add-media-image.svg';
import Skeleton, { SkeletonTheme } from 'react-loading-skeleton';
import 'react-loading-skeleton/dist/skeleton.css';

import Tag from './tag';
import VirtualizedTagGrid from './virtualized-tag-grid';
import TagListView from './tag-list-view';
import NewTagModal from './new-tag-modal';
import NewCategoryModal from './new-category-modal';
import PeopleGrid, {
  PEOPLE_CATEGORY,
  PersonEditModal,
} from './people-grid';
import './taxonomy.css';
import Category from './category';
import SuggestionSections from './suggestion-sections';
import { invoke } from '../../platform';
import QueryInput from '../query-input/QueryInput';
import { useTagSearch } from '../../hooks/useTagSearch';
import { useMeaningMode } from '../../hooks/useMeaningMode';

const VIRTUALIZE_THRESHOLD = 300;

type Concept = {
  label: string;
  category: string;
  weight: number;
  // Optional so the shared useTagSearch results (TagConcept, description?:) are
  // assignable to Concept-typed row renderers.
  description?: string;
};

type TagViewMode = 'card' | 'list';

type Category = {
  label: string;
  weight: number;
  description: string;
  tagViewMode?: TagViewMode;
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

  // The unified query (chips) shown in QueryInput.
  const query = useSelector(libraryService, (state) => state.context.query);

  const state = useSelector(
    libraryService,
    (state) => state,
    (a, b) => {
      return a.matches(b);
    }
  );

  // Two-tier search state: tagFilterInput updates on every keystroke so the
  // input feels responsive; tagFilter is the debounced value dispatched to the
  // search worker. The actual fuzzy match runs off-thread (see the worker
  // setup below), so the debounce only needs to coalesce bursts of keystrokes
  // rather than guard the main thread — hence a short 150ms window.
  const [tagFilterInput, setTagFilterInput] = useState<string>('');
  const [tagFilter, setTagFilter] = useState<string>('');
  // "Search by meaning" mode: typed text commits as a visual: (text→image)
  // predicate instead of filtering the tag tree. Shared + sticky with the
  // command palette (useMeaningMode) — stays on until the user toggles it off.
  const { meaningMode } = useMeaningMode();
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
    // In meaning mode the text drives a semantic query, not tag-tree filtering
    // — drop any pending/active filter. (The toggle may flip from either this
    // sidebar or the command palette; both arrive here via useMeaningMode.)
    if (meaningMode) {
      setTagFilter('');
      debouncedSetTagFilter.current.cancel();
      return;
    }
    debouncedSetTagFilter.current(tagFilterInput);
  }, [tagFilterInput, meaningMode]);
  useEffect(() => {
    const debounced = debouncedSetTagFilter.current;
    return () => {
      debounced.cancel();
    };
  }, []);

  const [addingTag, setAddingTag] = useState<boolean>(false);
  const [addingPerson, setAddingPerson] = useState<boolean>(false);
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
  // All three taxonomy fetches are gated on initSessionId, which the state
  // machine only assigns once it reaches its post-DB `init` state. The taxonomy
  // sidebar mounts during boot, so without this gate these queries fire before
  // load-db registers their IPC handlers and fail with "No handler registered"
  // on every launch (the fetch then succeeds on retry, but the error is logged).
  const { data: categories } = useQuery<Category[], Error>(
    ['taxonomy', 'categories', initSessionId],
    loadCategories,
    { enabled: !!initSessionId, staleTime: Infinity }
  );

  const { data: activeCategoryTags, isFetching: isFetchingCategoryTags } =
    useQuery<Concept[], Error>(
      ['taxonomy', 'category-tags', activeCategory, initSessionId],
      () => loadCategoryTags(activeCategory),
      { enabled: !!activeCategory && !!initSessionId, staleTime: Infinity }
    );

  // Warmed as soon as the search input gains focus (or, as a fallback, on the
  // first keystroke) so the fetch + worker indexing overlap with the user
  // composing their query — by the time they type, the data is usually already
  // cached and indexed, so the initial search isn't stalled on the network.
  const { data: allTagsData, isFetching: isFetchingAllTags } = useQuery<
    Concept[],
    Error
  >(['taxonomy', 'all-tags', initSessionId], loadAllTags, {
    enabled: (searchFocused || !!tagFilterInput) && !!initSessionId,
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

  // Tag search now runs through the shared, pre-warmed singleton index (one
  // worker for the whole app) via useTagSearch. We pass the already-debounced
  // tagFilter; the hook debounces again (harmless) and returns ranked,
  // pre-capped matches across all categories.
  const { results: searchResults } = useTagSearch(tagFilter, !!tagFilter);

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
            query={query}
            textValue={tagFilterInput}
            onTextChange={setTagFilterInput}
            filteringMode={filteringMode}
            onCycleFilterMode={() =>
              libraryService.send({
                type: 'CHANGE_SETTING',
                data: { filteringMode: getNextFilterMode(filteringMode) },
              })
            }
            onFocus={() => setSearchFocused(true)}
            onSubmitText={() => {
              // Commit the top tag suggestion as a predicate, then clear text.
              const top = searchResults[0];
              if (top) {
                libraryService.send({
                  type: 'ADD_PREDICATE',
                  data: {
                    predicate: {
                      type: 'tag',
                      value: top.label,
                      exclude: false,
                      join: filteringMode === 'OR' ? 'OR' : 'AND',
                    },
                  },
                });
                setTagFilterInput('');
                setTagFilter('');
                debouncedSetTagFilter.current.cancel();
              }
            }}
            onRemovePredicate={(key) =>
              libraryService.send({
                type: 'REMOVE_PREDICATE',
                data: { key },
              })
            }
            onToggleExclude={(key) =>
              libraryService.send({
                type: 'TOGGLE_EXCLUDE',
                data: { key },
              })
            }
            onSetPredicateJoin={(key, join) =>
              libraryService.send({
                type: 'SET_PREDICATE_JOIN',
                data: { key, join },
              })
            }
            onUpdatePredicateBlend={(key, patch) =>
              libraryService.send({
                type: 'UPDATE_PREDICATE_BLEND',
                data: { key, patch },
              })
            }
            onSubmitVisual={(t) => {
              libraryService.send({
                type: 'ADD_PREDICATE',
                data: {
                  predicate: {
                    type: 'visual',
                    value: t,
                    exclude: false,
                    join: filteringMode === 'OR' ? 'OR' : 'AND',
                  },
                },
              });
              setTagFilterInput('');
              setTagFilter('');
              debouncedSetTagFilter.current.cancel();
            }}
            onClearText={() => {
              setTagFilterInput('');
              setTagFilter('');
              debouncedSetTagFilter.current.cancel();
              setEditingTag(null);
            }}
            onClearAll={() => {
              libraryService.send({ type: 'CLEAR_QUERY' });
              setTagFilterInput('');
              setTagFilter('');
              debouncedSetTagFilter.current.cancel();
              setEditingTag(null);
            }}
            disabled={isDisabled}
          />
        </div>
        {/* Body row: fills the height left under the search bar so the inner
            scroll areas size against that remaining height (not the whole
            panel) and keep their full bottom padding — otherwise their bottoms
            overhang the panel and the last row hides under the bottom overlay. */}
        <div className="taxonomy-body">
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
          <div
            className={`new-tag`}
            onClick={() =>
              // People is face-managed: "+" creates a person (which creates
              // its tag through the server bridge), not a bare tag — a bare
              // tag in People would have no person row behind it.
              activeCategory === PEOPLE_CATEGORY
                ? setAddingPerson(true)
                : setAddingTag(true)
            }
          >
            <div className="tag-label">+</div>
          </div>
        )}
        {(() => {
          // The search-results area holds two panes: the tag results (cards /
          // list) on the left and the SuggestionSections column (categories /
          // paths / description / hash) on the right. taxonomy.css drives the
          // responsive two-column ⇄ stacked behaviour via a container query on
          // `.search-results`; here we only decide which panes exist so empty
          // regions collapse instead of reserving blank space.
          const resultsEl = (() => {
            if (!(activeCategory || tagFilter)) {
              return <div className={`tags`} />;
            }
            // People is a special case of a tag category: person names are
            // real tags (kept in sync server-side), but browsing the category
            // shows person cards with face crops and person-aware management
            // (rename/merge/delete through /api/people). Search results still
            // show People tags as ordinary tag cards, which behave correctly.
            if (!tagFilter && activeCategory === PEOPLE_CATEGORY) {
              return <PeopleGrid isDisabled={isDisabled} />;
            }
            // Search results span categories — always use card style.
            // For an active category, honour its persisted tagViewMode.
            const activeViewMode: TagViewMode = tagFilter
              ? 'card'
              : (categoriesByLabel[activeCategory]
                  ?.tagViewMode as TagViewMode) || 'card';
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
            // Searching with zero tag matches: collapse the results pane
            // entirely (return null) so the suggestions column can take the
            // full width instead of sitting beside an empty grid.
            if (tagFilter && tags.length === 0) {
              return null;
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
          })();

          const showSuggestions = !!tagFilter;
          const hasResults = resultsEl !== null;
          const wrapperClass = [
            'search-results',
            hasResults ? 'has-results' : 'no-results',
            showSuggestions ? 'has-suggestions' : '',
          ]
            .filter(Boolean)
            .join(' ');

          // The outer .search-results-container exists only to establish the
          // container-query context: a container query can style a container's
          // descendants but not the container element itself, so the
          // stacked⇄two-column switch must live on the inner .search-results.
          return (
            <div className="search-results-container">
              <div className={wrapperClass}>
                {hasResults && (
                  <div className="results-pane">{resultsEl}</div>
                )}
                {showSuggestions && (
                  <div className="suggestions-pane">
                    <SuggestionSections
                      text={tagFilter}
                      categories={categories ?? []}
                      onAdd={(predicate) =>
                        libraryService.send({
                          type: 'ADD_PREDICATE',
                          data: {
                            predicate: {
                              ...predicate,
                              join: filteringMode === 'OR' ? 'OR' : 'AND',
                            },
                          },
                        })
                      }
                    />
                  </div>
                )}
              </div>
            </div>
          );
        })()}
        </div>
      </div>
      {activeCategory && addingTag ? (
        <NewTagModal
          categoryLabel={activeCategory}
          handleClose={() => setAddingTag(false)}
        />
      ) : null}
      {addingPerson ? (
        <PersonEditModal
          person={null}
          people={[]}
          handleClose={() => setAddingPerson(false)}
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
