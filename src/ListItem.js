import React, { useRef } from "react";
import useScrollOnDrag from "react-scroll-ondrag";

const url = window.require("url");
const path = window.require("path");

import { SIZE, EXTENSIONS, VIEW } from "./constants";

function Detail({ fileName, size, handleClick, runScroll, controlMode }) {
  const containerRef = useRef(null);
  const { events } = useScrollOnDrag(containerRef);
  return (
    <div
      className="listContainer"
      onDoubleClick={handleClick}
      onScroll={e => {
        e.preventDefault();
        e.stopPropagation();
      }}
      tabIndex="0"
      {...events}
      ref={containerRef}
    >
      {EXTENSIONS.img.includes(path.extname(fileName).toLowerCase()) && (
        <img
          key={fileName}
          src={url.format({
            protocol: "file",
            pathname: fileName
          })}
          className="listImage"
        />
      )}
      {EXTENSIONS.video.includes(path.extname(fileName).toLowerCase()) && (
        <video
          className="listImage"
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
