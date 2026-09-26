import { useContext, useRef, useState, useEffect } from 'react';
import { useDrop } from 'react-dnd';
import { useSelector } from '@xstate/react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import ConfirmDeleteCategory from './confirm-delete-category';
import editPencil from '../../../../assets/edit-pencil.svg';
import deleteIcon from '../../../../assets/delete.svg';
import { invoke } from '../../platform';
import { GlobalStateContext } from '../../state';

// Which glyph marks a system category in the list.
export type SystemCategoryIcon = 'suggested' | 'people' | 'duplicates';

type Category = {
  label: string;
  // Client-only entries (the Duplicates panel) are browsable but not
  // editable: no rename/delete, no drop-to-move, no category context menu.
  synthetic?: boolean;
  // Machine-managed (Suggested, People, Duplicates): styled a notch quieter,
  // never renamed or deleted from here (the app owns their names and their
  // rows), and marked with an icon.
  system?: boolean;
  icon?: SystemCategoryIcon;
};

// Inline so the glyphs inherit the row's text colour and need no asset
// pipeline: sparkles (machine suggestions), a person, two overlapping frames.
const SYSTEM_ICONS: Record<SystemCategoryIcon, JSX.Element> = {
  suggested: (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z" />
      <path d="M19 16l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z" />
      <path d="M4.5 15l.5 1.5 1.5.5-1.5.5-.5 1.5-.5-1.5L2.5 17l1.5-.5z" />
    </svg>
  ),
  people: (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="8" r="4" />
      <path d="M4 21c0-4 3.6-6.5 8-6.5s8 2.5 8 6.5" />
    </svg>
  ),
  duplicates: (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="3" y="3" width="12" height="12" rx="2" />
      <rect x="9" y="9" width="12" height="12" rx="2" />
    </svg>
  ),
};

type Props = {
  category: Category;
  activeCategory: string | null;
  setActiveCategory: (category: string) => void;
  handleEditAction: (category: string) => void;
};

const moveTag = async ({
  tag,
  category,
}: {
  tag: string;
  category: string;
}) => {
  console.log('move', tag, category);
  await invoke('move-tag', [tag, category]);
};

export default function Category({
  category,
  activeCategory,
  setActiveCategory,
  handleEditAction,
}: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  // View-only public visitors: no rename/delete, no drop-to-move-tag.
  // Category click (browse) stays.
  const canWrite = useSelector(
    libraryService,
    (state) => state.context.canWrite
  );
  const ref = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const { mutate } = useMutation({
    mutationFn: moveTag,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      setActiveCategory(category.label);
    },
  });

  const [collectedProps, drop] = useDrop(
    () => ({
      accept: ['TAG'],
      canDrop: () => canWrite && !category.synthetic,
      collect: (monitor) => ({
        isOver: monitor.isOver(),
      }),
      drop: (droppedTag: any, monitor) => {
        if (!canWrite) return;
        mutate({
          tag: droppedTag.label,
          category: category.label,
        });
      },
    }),
    [category, canWrite]
  );

  drop(ref);
  return (
    <div
      ref={ref}
      key={category.label}
      className={`category ${activeCategory === category.label && 'active'} ${
        collectedProps.isOver ? 'hovered' : ''
      }${category.system ? ' system' : ''}`}
      onClick={() => setActiveCategory(category.label)}
      onContextMenu={(e) => {
        if (e.shiftKey && !category.synthetic) {
          e.preventDefault();
          e.stopPropagation();
          libraryService.send('SHOW_CONTEXT_PALETTE', {
            position: { x: e.clientX, y: e.clientY },
            target: { type: 'category', category: category.label },
          });
        }
      }}
    >
      <div
        className="category-label"
        title={
          category.system
            ? 'Managed by the app — cannot be renamed or deleted'
            : undefined
        }
      >
        {category.icon && (
          <span className="category-icon">{SYSTEM_ICONS[category.icon]}</span>
        )}
        {category.label}
      </div>
      {canWrite && !category.synthetic && !category.system && (
        <div className="actions">
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleEditAction(category.label);
            }}
          >
            <img src={editPencil} />
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowDeleteModal(true);
            }}
          >
            <img src={deleteIcon} />
          </button>
        </div>
      )}
      {showDeleteModal && canWrite ? (
        <ConfirmDeleteCategory
          handleClose={() => setShowDeleteModal(false)}
          currentValue={category.label}
        />
      ) : null}
    </div>
  );
}
