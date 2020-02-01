import React, { useState, useEffect, useRef } from "react";
const electron = window.require("electron");
var shuffle = window.require("shuffle-array");
import ReactDOM from "react-dom";
import "babel-polyfill";
import "./style.css";
import HotKeyController from "./HotKeyController";
import { getFolder } from "./fsTools";

import Detail from "./Detail";
import List from "./List";
import HotCorner from "./HotCorner";
import SettingsButton from "./SettingsButton";
import Spinner from "./Spinner";
// NODE IMPORTS
const settings = window.require("electron-settings");
const atob = window.require("atob");

import {
  SORT,
  FILTER,
  SIZE,
  VIEW,
  CONTROL_MODE,
  getNext,
  LIST_SIZE
} from "./constants";
import loadImageList from "./loadImageList";
import Status from "./Status";

function App() {
  const [view, setView] = useState(VIEW.DETAIL);
  const [filePath, setPath] = useState(atob(window.location.search.substr(1)));
  const [loading, setLoading] = useState(false);
  const [shuffles, setShuffles] = useState(true);

  const [status, setStatus] = useState(false);
  const [items, setItems] = useState([]);
  const [cursor, setCursor] = useState(0);
  const [controlMode, setControlMode] = useState(
    CONTROL_MODE[settings.get("settings.controlMode")]
  );

  const [sort, setSort] = useState(SORT[settings.get("settings.defaultSort")]);
  const [filter, setFilter] = useState(FILTER.ALL);
  const [size, setSize] = useState(SIZE.OVERSCAN);
  const [listSize, setListSize] = useState(LIST_SIZE.OVERSCAN);

  const [tall, setTall] = useState(true);

  const [recursive, setRecursive] = useState(false);

  function changePath() {
    electron.remote.dialog
      .showOpenDialog(electron.remote.getCurrentWindow(), ["openFile"])
      .then(files => {
        console.log(files);
        if (files.filePaths.length > 0) {
          setPath(files.filePaths[0]);
        }
      });
  }
  // Initialize State from settings.
  useEffect(() => {
    if (settings.has("settings.scaleMode")) {
      setSize(SIZE[settings.get("settings.scaleMode")]);
    }
  }, []);
  // Reload data from image provider if directory, filter, or recursive setting changes
  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      const data = await loadImageList(
        getFolder(filePath),
        filter.value,
        sort,
        recursive
      );
      setItems(data.items);
      let cursor = data.items.findIndex(item => {
        return item.fileName === filePath;
      });
      if (cursor < 0) {
        cursor = 0;
      }
      setCursor(cursor);
      setLoading(false);
    }
    if (filePath.length > 1) {
      fetchData();
    } else {
      changePath();
    }
  }, [filePath, sort, filter, recursive]);

  function handleClick(e) {
    e.preventDefault();
    // If click is on the left decrease the cursor, if it is on the left increase it.
    e.pageX > window.innerWidth / 2
      ? setCursor(cursor === items.length - 1 ? cursor : cursor + 1)
      : setCursor(cursor === 0 ? cursor : cursor - 1);
  }

  function handleScroll(e) {
    e.preventDefault();
    // If delta y is positive increase cursor value, otherwise decrease.
    e.deltaY > 0
      ? setCursor(cursor === items.length - 1 ? cursor : cursor + 1)
      : setCursor(cursor === 0 ? cursor : cursor - 1);
  }

  function handleKeyPress(e) {
    switch (e.key) {
      case "s":
        e.preventDefault();
        setSort(getNext(SORT, sort));
        break;
      case "c":
        e.preventDefault();
        setSize(getNext(SIZE, size.key));

        break;
      case "m":
        e.preventDefault();
        setControlMode(getNext(CONTROL_MODE, controlMode));

        break;
      case "r":
        e.preventDefault();
        setRecursive(!recursive);
        break;
      case "t":
        e.preventDefault();
        setTall(!tall);
        break;
      case "g":
        e.preventDefault();
        setFilter(FILTER.GIF);
        break;
      case "a":
        e.preventDefault();
        setFilter(FILTER.ALL);
        break;
      case "v":
        e.preventDefault();
        setFilter(FILTER.VIDEO);
        break;
      case "x":
        e.preventDefault();
        setItems(shuffle(items));
        setShuffles(!shuffles);
        break;
      case "m":
        e.preventDefault();
        setFilter(FILTER.VIDEO);
        break;
      case "j":
        e.preventDefault();
        setFilter(FILTER.STATIC);
        break;
      default:
        e.preventDefault();
        console.log(`pressed ${e.key}`);
    }
  }

  if (loading) {
    return (
      <div
        tabIndex="0"
        onKeyPress={handleKeyPress}
        className="loadingContainer"
      >
        <Spinner />
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <React.Fragment>
        <HotKeyController handleKeyPress={handleKeyPress} />
        <SettingsButton handleClick={() => setStatus(!status)} />
        <div className="dragArea"></div>
        {status && (
          <Status
            status={{
              filePath,
              sort,
              filter,
              size,
              listSize,
              controlMode,
              recursive,
              items
            }}
            controls={{
              changePath,
              setSort,
              setFilter,
              setSize,
              setListSize,
              setControlMode,
              setRecursive
            }}
          />
        )}
        <div
          tabIndex="0"
          onKeyPress={handleKeyPress}
          className="noItemsContainer"
        >
          <span className="noItemsMessage">No Images Found</span>
        </div>
      </React.Fragment>
    );
  }

  return (
    <React.Fragment>
      <HotKeyController handleKeyPress={handleKeyPress} />

      <SettingsButton handleClick={() => setStatus(!status)} />
      <div className="dragArea"></div>
      {status && (
        <Status
          status={{
            filePath,
            sort,
            filter,
            size,
            listSize,
            controlMode,
            recursive,
            items
          }}
          controls={{
            changePath,
            setSort,
            setFilter,
            setSize,
            setListSize,
            setControlMode,
            setRecursive
          }}
        />
      )}

      {view === VIEW.DETAIL ? (
        <React.Fragment>
          <HotCorner handleClick={() => setView(VIEW.LIST)} />
          <Detail
            fileName={items[cursor].fileName}
            size={size}
            handleClick={handleClick}
            handleScroll={handleScroll}
            controlMode={controlMode}
          />
        </React.Fragment>
      ) : (
        <React.Fragment>
          <List
            filter={filter}
            fileList={items}
            shuffles={shuffles}
            size={listSize}
            tall={tall}
            cursor={cursor}
            controlMode={controlMode}
            setPath={setPath}
            handleClick={i => {
              setCursor(i);
              setView(VIEW.DETAIL);
            }}
          />
        </React.Fragment>
      )}
    </React.Fragment>
  );
}
ReactDOM.render(<App />, document.getElementById("root"));
