import { useState, useContext, useRef } from 'react';
import { useSelector } from '@xstate/react';
import { useQuery } from '@tanstack/react-query';
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
import NewTagModal from './new-tag-modal';
import NewCategoryModal from './new-category-modal';
import './taxonomy.css';
import Category from './category';

type Concept = {
  label: string;
  category: string;
  weight: number;
};

type Category = {
  label: string;
  tags: Concept[];
};

type Taxonomy = {
  [key: string]: Category;
};

type FilterModeIconMap = {
  [key in FilterModeOption]: string;
};

async function loadTaxonomy(): Promise<Taxonomy> {
  const taxonomy = await window.electron.ipcRenderer.invoke(
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
  const { applyTagPreview, filteringMode, applyTagToAll } = useSelector(
    libraryService,
    (state) => {
      return state.context.settings;
    }
  );

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

  function setTextFilter(text: string) {
    libraryService.send({
      type: 'SET_TEXT_FILTER',
      data: { textFilter: text },
    });
  }

  const [tagFilter, setTagFilter] = useState<string>('');

  const [addingTag, setAddingTag] = useState<boolean>(false);
  const [addingCategory, setAddingCategory] = useState<boolean>(false);
  const [editingTag, setEditingTag] = useState<string | null>(null);
  const [editingCategory, setEditingCategory] = useState<string | null>(null);
  const activeCategory = useSelector(
    libraryService,
    (state) => state.context.activeCategory
  );
  function setActiveCategory(category: string) {
    libraryService.send({
      type: 'SET_ACTIVE_CATEGORY',
      data: { category },
    });
  }
  const { data: taxonomy } = useQuery<Taxonomy, Error>(
    ['taxonomy', initSessionId],
    loadTaxonomy
  );

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
  console.log(taxonomy);
  const tags = Object.values(taxonomy)
    .reduce((acc, category) => {
      return [...acc, ...category.tags];
    }, [] as Concept[])
    .filter((tag) => {
      if (tagFilter) {
        return (
          tag.label && tag.label.toLowerCase().includes(tagFilter.toLowerCase())
        );
      } else {
        return tag.category && tag.category === activeCategory;
      }
    })
    .sort((a, b) => {
      return a.weight - b.weight;
    });
  return (
    <>
      <div className={`Taxonomy`}>
        <div className="search">
          <div className="textSearch">
            <input
              type="text"
              placeholder="Search Content"
              value={textFilter}
              onChange={(e) => setTextFilter(e.currentTarget.value)}
            />
            <button
              onClick={() => {
                setTextFilter('');
                setEditingTag(null);
              }}
            >
              <img src={clear} />
            </button>
          </div>
          <div className="textSearch">
            <input
              type="text"
              placeholder="Search Tags"
              value={tagFilter}
              onChange={(e) => setTagFilter(e.currentTarget.value)}
            />
            <button
              onClick={() => {
                setTagFilter('');
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
                  setActiveCategory={setActiveCategory}
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
        <div className={`tags`}>
          {activeCategory || tagFilter
            ? tags.map((tag: Concept) => {
                return (
                  <Tag
                    tags={tags}
                    tag={{
                      label: tag.label,
                      weight: tag.weight,
                      category: tag.category,
                    }}
                    active={selectedTags.includes(tag.label)}
                    handleEditAction={setEditingTag}
                    key={tag.label}
                  />
                );
              })
            : null}
        </div>
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
        />
      ) : null}
      {editingCategory ? (
        <NewCategoryModal
          handleClose={() => setEditingCategory(null)}
          setCategory={setActiveCategory}
          currentValue={editingCategory}
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
