// Swallow the `click` that trails the current mousedown, in the capture
// phase, so it never reaches React tree handlers.
//
// Used when a mousedown-driven "click outside closes this" surface (command
// palette, context palette) dismisses itself: in trackpad/touchpad mode the
// detail view binds its own onClick (cursor advance/decrement), so without
// intervention the user's click both closes the palette AND lands on the
// detail handler, jumping to a different media item.
export function absorbNextClick(timeoutMs = 300): void {
  const absorbClick = (clickEvent: Event) => {
    clickEvent.stopPropagation();
    clickEvent.preventDefault();
  };
  window.addEventListener('click', absorbClick, {
    capture: true,
    once: true,
  });
  // Safety net — if no click follows the mousedown (e.g. the user releases
  // off-window), drop the listener so it can't swallow a future legitimate
  // click.
  setTimeout(
    () =>
      window.removeEventListener('click', absorbClick, {
        capture: true,
      } as EventListenerOptions),
    timeoutMs
  );
}
