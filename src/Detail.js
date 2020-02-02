import React, { useRef } from "react";
import useScrollOnDrag from "react-scroll-ondrag";

const url = window.require("url");
const path = window.require("path");

import { EXTENSIONS } from "./constants";

function Detail({
  fileName,
  size,
  handleClick,
  handleScroll,
  controlMode,
  setPath
}) {
  const containerRef = useRef(null);
  const { events } = useScrollOnDrag(containerRef);
  return (
    <div
      className="container"
      onClick={controlMode === "MOUSE" ? null : handleClick}
      onContextMenu={() => setPath(fileName)}
      onWheel={controlMode === "MOUSE" ? handleScroll : null}
      tabIndex="0"
      {...events}
      ref={controlMode === "MOUSE" ? containerRef : null}
    >
      {EXTENSIONS.img.includes(path.extname(fileName).toLowerCase()) && (
        <img
          key={fileName}
          src={url.format({
            protocol: "file",
            pathname: fileName
          })}
          className={size.className}
        />
      )}
      {EXTENSIONS.video.includes(path.extname(fileName).toLowerCase()) && (
        <video
          className={size.className}
          src={url.format({
            protocol: "file",
            pathname: fileName
          })}
          loop
          autoPlay
          muted
          controls
        />
      )}
    </div>
  );
}

export default Detail;
