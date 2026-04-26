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
  tags: Concept[];
  description: string;
  tagViewMode?: TagViewMode;
};

type Taxonomy = {
  [key: string]: Category;
};

type FilterModeIconMap = {
  [key in FilterModeOption]: string;
};

async function loadTaxonomy(): Promise<Taxonomy> {
  const taxonomy = await invoke(
    'load-taxonomy',
    []
  );
  return taxonomy as Taxonomy;
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
  // input feels responsive; tagFilter is the debounced value that drives the
  // (potentially expensive) filter+sort over many tags.
  const [tagFilterInput, setTagFilterInput] = useState<string>('');
  const [tagFilter, setTagFilter] = useState<string>('');
  const debouncedSetTagFilter = useRef(
    debounce((value: string) => {
      setTagFilter(value);
    }, 300)
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
  const { data: taxonomy } = useQuery<Taxonomy, Error>(
    ['taxonomy', initSessionId],
    loadTaxonomy
  );

  // Ensure activeCategory is valid; if not, reset to the first available category.
  // Suspended while a search is active — during search no category should be selected.
  useEffect(() => {
    if (!taxonomy) return;
    if (tagFilter) return;
    const categories = Object.values(taxonomy || {});
    if (categories.length === 0) return;
    const exists = categories.some((c) => c.label === activeCategory);
    if (!exists) {
      setActiveCategory(categories[0].label);
    }
  }, [taxonomy, activeCategory, tagFilter]);

  // When a search becomes active, clear the active category — the user is now
  // searching across categories, not viewing one.
  useEffect(() => {
    if (tagFilter) {
      setActiveCategory('');
    }
  }, [tagFilter]);

  // Given every Category is 20 pixels tall, this will scroll to the active category
  // when it is not visible in the list by setting the scrollTop of the categoryListRef

  if (categoryListRef.current && activeCategory) {
    const activeCategoryIndex = Object.values(taxonomy || {}).findIndex(
      (category) => category.label === activeCategory
    );
    if (activeCategoryIndex > -1) {
      const activeCategoryTop =
        activeCategoryIndex * 25.5 - categoryListRef.current.clientHeight / 2;
      categoryListRef.current.scrollTop = activeCategoryTop;
    }
  }

  // Flat list of every tag in the taxonomy. Built once per taxonomy load
  // so the Fuse index can be reused across keystrokes. Defensive filter
  // drops tags without a label — Fuse and the row components both assume
  // a non-empty string and would otherwise throw when one slips in.
  const allTags = useMemo(() => {
    if (!taxonomy) return [] as Concept[];
    return Object.values(taxonomy).reduce((acc, category) => {
      for (const t of category.tags) {
        if (t && typeof t.label === 'string' && t.label.length > 0) {
          acc.push(t);
        }
      }
      return acc;
    }, [] as Concept[]);
  }, [taxonomy]);

  const fuse = useMemo(
    () =>
      new Fuse(allTags, {
        keys: ['label'],
        threshold: 0.4,
        ignoreLocation: true,
        minMatchCharLength: 1,
      }),
    [allTags]
  );

  const tags = useMemo(() => {
    if (!taxonomy) return [] as Concept[];
    if (tagFilter) {
      // Fuzzy match across all categories. Fuse returns results pre-sorted
      // by relevance (best match first). Cap at MAX_SEARCH_RESULTS so a
      // pathological query can't render thousands of cards at once.
      return fuse
        .search(tagFilter, { limit: MAX_SEARCH_RESULTS })
        .map((r) => r.item);
    }
    return allTags
      .filter((tag) => tag.category && tag.category === activeCategory)
      .sort((a, b) => a.weight - b.weight);
  }, [taxonomy, tagFilter, activeCategory, allTags, fuse]);

  if (!taxonomy || state.matches('loadingDB') || state.matches('selectingDB')) {
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
            {(Object.values(taxonomy) || []).map((category) => {
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
            : (taxonomy?.[activeCategory]?.tagViewMode as TagViewMode) || 'card';
          // Reordering by drag is meaningless when results are sorted by
          // search relevance — disable DnD while a search is active.
          const disableReorder = !!tagFilter;
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
      {activeCategory && editingTag ? (
        <NewTagModal
          categoryLabel={activeCategory}
          handleClose={() => setEditingTag(null)}
          currentValue={editingTag}
          currentDescription={
            taxonomy?.[activeCategory]?.tags?.find(
              (t: Concept) => t.label === editingTag
            )?.description || ''
          }
        />
      ) : null}
      {editingCategory ? (
        <NewCategoryModal
          handleClose={() => setEditingCategory(null)}
          setCategory={setActiveCategory}
          currentValue={editingCategory}
          currentDescription={
            taxonomy?.[editingCategory]?.description || ''
          }
          currentTagViewMode={
            (taxonomy?.[editingCategory]?.tagViewMode as TagViewMode) || 'card'
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
