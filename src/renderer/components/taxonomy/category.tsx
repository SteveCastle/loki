import { useRef, useState, useEffect } from 'react';
import { useDrop } from 'react-dnd';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import ConfirmDeleteCategory from './confirm-delete-category';
import editPencil from '../../../../assets/edit-pencil.svg';
import deleteIcon from '../../../../assets/delete.svg';

type Concept = {
  label: string;
  category: string;
  weight: number;
};

type Category = {
  label: string;
  tags: Concept[];
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
  await window.electron.ipcRenderer.invoke('move-tag', [tag, category]);
};

export default function Category({
  category,
  activeCategory,
  setActiveCategory,
  handleEditAction,
}: Props) {
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
      collect: (monitor) => ({
        isOver: monitor.isOver(),
      }),
      drop: (droppedTag: any, monitor) => {
        mutate({
          tag: droppedTag.label,
          category: category.label,
        });
      },
    }),
    [category]
  );

  drop(ref);
  return (
    <div
      ref={ref}
      key={category.label}
      className={`category ${activeCategory === category.label && 'active'} ${
        collectedProps.isOver ? 'hovered' : ''
      }`}
      onClick={() => setActiveCategory(category.label)}
    >
      <div className="category-label">{category.label}</div>
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
      {showDeleteModal ? (
        <ConfirmDeleteCategory
          handleClose={() => setShowDeleteModal(false)}
          currentValue={category.label}
        />
      ) : null}
    </div>
  );
}
