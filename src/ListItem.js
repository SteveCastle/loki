import React, { useRef, useLayoutEffect, useState } from "react";
import useScrollOnDrag from "react-scroll-ondrag";

const url = window.require("url");
const path = window.require("path");

import { SIZE, EXTENSIONS, VIEW } from "./constants";

function ListItem({ fileName, handleClick }) {
  const [isPortrait, setPortrait] = useState(false);

  const [loaded, setLoaded] = useState(false);
  const containerRef = useRef(null);
  const imageRef = useRef(null);

  const { events } = useScrollOnDrag(containerRef);
  useLayoutEffect(() => {
    if (loaded) {
      if (imageRef.current.offsetHeight > imageRef.current.offsetWidth) {
        setPortrait(true);
      }
      const verticalCenter =
        (imageRef.current.offsetHeight - containerRef.current.offsetHeight) / 2;
      const horizontalCenter =
        (imageRef.current.offsetWidth - containerRef.current.offsetWidth) / 2;

      containerRef.current.scrollTo(horizontalCenter, verticalCenter);
    }
  }, [loaded]);
  return (
    <div
      className="listContainer"
      onDoubleClick={handleClick}
      tabIndex="0"
      {...events}
      ref={containerRef}
    >
      {EXTENSIONS.img.includes(path.extname(fileName).toLowerCase()) && (
        <img
          onLoad={() => setLoaded(true)}
          key={fileName}
          ref={imageRef}
          src={url.format({
            protocol: "file",
            pathname: fileName
          })}
          className={`${
            isPortrait ? "listImagePortrait" : "listImageLandscape"
          }`}
        />
      )}
      {EXTENSIONS.video.includes(path.extname(fileName).toLowerCase()) && (
        <video
          className={`${
            isPortrait ? "listVideoPortrait" : "listVideoLandscape"
          }`}
          onLoadStart={() => setLoaded(true)}
          ref={imageRef}
          src={url.format({
            protocol: "file",
            pathname: fileName
          })}
          loop
          autoPlay
          muted
        />
      )}
    </div>
  );
}

export default ListItem;
