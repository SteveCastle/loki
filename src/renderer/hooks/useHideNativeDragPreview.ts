import { useEffect } from 'react';
import type { ConnectDragPreview } from 'react-dnd';
import { getEmptyImage } from 'react-dnd-html5-backend';

/**
 * Replaces the HTML5 backend's native drag ghost (a translucent snapshot of
 * the entire source DOM element) with an empty image, so the custom
 * <DragChipLayer /> (components/controls/drag-layer.tsx) can render a
 * compact cursor-following chip instead. Connect the `preview` ref returned
 * by useDrag through this on every drag source that should use the chip.
 */
export default function useHideNativeDragPreview(preview: ConnectDragPreview) {
  useEffect(() => {
    preview(getEmptyImage(), { captureDraggingState: true });
  }, [preview]);
}
