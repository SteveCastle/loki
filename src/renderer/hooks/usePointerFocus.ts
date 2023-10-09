import {useRef, useEffect } from "react";

function useFocusOnMouseEnter() {
  const elementRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!elementRef.current) {
      return;
    }

    const handleMouseEnter = () => {
      elementRef.current?.focus();
    };

    const element = elementRef.current;
    element.addEventListener("mouseenter", handleMouseEnter);

    return () => {
      element.removeEventListener("mouseenter", handleMouseEnter);
    };
  }, [elementRef]);
  return elementRef
}

export default useFocusOnMouseEnter;
