import React, { useRef, useState, useEffect } from "react";
import useScrollOnDrag from "react-scroll-ondrag";

const url = window.require("url");
const path = window.require("path");

import { EXTENSIONS } from "./constants";

function useWindowSize() {
  const [windowSize, setWindowSize] = useState({
    width: undefined,
    height: undefined,
  });
  useEffect(() => {
    function handleResize() {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    }
    window.addEventListener("resize", handleResize);
    handleResize();
    return () => window.removeEventListener("resize", handleResize);
  }, []);
  return windowSize;
}

function Detail({
  items,
  cursor,
  size,
  audio,
  volume,
  videoControls,
  setAudio,
  setVolume,
  handleClick,
  handleScroll,
  controlMode,
  handleDoubleClick,
  handleRightClick,
}) {
  const containerRef = useRef(null);
  const imageRef = useRef(null);
  const { events } = useScrollOnDrag(containerRef);
  const { width, height } = useWindowSize();
  const previousFileName = items[cursor - 1]?.fileName;
  const fileName = items[cursor].fileName;
  const nextFileName = items[cursor + 1]?.fileName;

  const [isPortrait, setPortrait] = useState(false);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    if (
      imageRef.current.offsetWidth / imageRef.current.offsetHeight >
      width / height
    ) {
      setPortrait(true);
    } else {
      setPortrait(false);
    }
    setLoaded(false);
  }, [loaded, size, fileName, width, height]);

  return (
    <div
      className={
        controlMode.key === "MOUSE" ? "container lock-scroll" : "container"
      }
      onClick={controlMode.key === "MOUSE" ? null : handleClick}
      onContextMenu={handleRightClick}
      onDoubleClick={handleDoubleClick}
      onWheel={controlMode.key === "MOUSE" ? handleScroll : null}
      tabIndex="0"
      {...events}
      ref={controlMode.key === "MOUSE" ? containerRef : null}
    >
      {EXTENSIONS.img.includes(path.extname(fileName).toLowerCase()) && (
        <>
          {previousFileName && (
            <img
              key={previousFileName}
              src={url.format({
                protocol: "file",
                pathname: previousFileName,
              })}
              className="previous"
            />
          )}
          <img
            ref={imageRef}
            key={fileName}
            onLoad={() => {
              setLoaded(true);
            }}
            src={url.format({
              protocol: "file",
              pathname: fileName,
            })}
            className={[
              size.className,
              isPortrait ? "portrait" : "landscape",
            ].join(" ")}
          />
          {nextFileName && (
            <img
              key={nextFileName}
              src={url.format({
                protocol: "file",
                pathname: nextFileName,
              })}
              className="next"
            />
          )}
        </>
      )}
      {EXTENSIONS.video.includes(path.extname(fileName).toLowerCase()) && (
        <video
          ref={imageRef}
          className={[
            size.className,
            isPortrait ? "portrait" : "landscape",
          ].join(" ")}
          src={url.format({
            protocol: "file",
            pathname: fileName,
          })}
          onPlaying={() => setLoaded(true)}
          loop
          autoPlay
          controls={videoControls}
          controlsList="nofullscreen nodownload noremoteplayback"
          muted={!audio}
          onLoadStart={(e) => (e.target.volume = volume)}
          onVolumeChange={(e) => {
            setVolume(e.target.volume);
            setAudio(!e.target.muted);
          }}
        />
      )}
    </div>
  );
}

export default Detail;
