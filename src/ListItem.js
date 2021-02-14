import React, { useRef, useEffect, useState } from "react";
import useScrollOnDrag from "react-scroll-ondrag";

const url = window.require("url");
const path = window.require("path");
import { EXTENSIONS } from "./constants";

function ListItem({ fileName, handleClick, size, handleRightClick }) {
  const [isPortrait, setPortrait] = useState(false);

  const [loaded, setLoaded] = useState(false);
  const containerRef = useRef(null);
  const imageRef = useRef(null);

  const { events } = useScrollOnDrag(containerRef);

  // Set scroll position of list item closer to middle top.
  useEffect(() => {
    if (loaded) {
      if (imageRef.current.offsetHeight > imageRef.current.offsetWidth) {
        setPortrait(true);
        setLoaded(false);
      } else {
        setPortrait(false);
      }
      const verticalCenter =
        (imageRef.current.scrollHeight - containerRef.current.offsetHeight) / 2;
      const horizontalCenter =
        (imageRef.current.scrollWidth - containerRef.current.offsetWidth) / 2;
      containerRef.current.scrollTo(horizontalCenter, verticalCenter);
    }
  });
  return (
    <React.Fragment>
      <div
        className="listContainer"
        {...events}
        onDoubleClick={handleClick}
        ref={containerRef}
      >
        {EXTENSIONS.img.includes(path.extname(fileName).toLowerCase()) && (
          <img
            onLoad={() => {
              setLoaded(true);
            }}
            key={fileName}
            ref={imageRef}
            src={url.format({
              protocol: "file",
              pathname: fileName,
            })}
            className={`${
              isPortrait
                ? `${size.className}ImagePortrait`
                : `${size.className}ImageLandscape`
            }`}
          />
        )}

        {EXTENSIONS.video.includes(path.extname(fileName).toLowerCase()) && (
          <video
            className={`${
              isPortrait
                ? `${size.className}VideoPortrait`
                : `${size.className}VideoLandscape`
            }`}
            onPlaying={() => setLoaded(true)}
            ref={imageRef}
            src={`${url.format({
              protocol: "file",
              pathname: fileName,
            })}#t=30,120`}
            loop
            autoPlay
            muted
          />
        )}
      </div>
    </React.Fragment>
  );
}

export default ListItem;
