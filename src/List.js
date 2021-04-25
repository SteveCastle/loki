import React, { Component } from "react";
import { FixedSizeGrid as Grid } from "react-window";
import HotKeyController from "./HotKeyController";

import ListItem from "./ListItem";
const Cell = ({ data, columnIndex, rowIndex, style }) =>
  data.fileList[rowIndex * data.columns + columnIndex] ? (
    <div
      tabIndex="0"
      style={{
        ...style,
        overflow: "hidden",
      }}
    >
      <ListItem
        className="listImage"
        size={data.size}
        useBucket={data.useBucket}
        handleSelection={() =>
          data.handleSelection(rowIndex * data.columns + columnIndex)
        }
        fileName={
          data.fileList[rowIndex * data.columns + columnIndex] &&
          data.fileList[rowIndex * data.columns + columnIndex].fileName
        }
      />
    </div>
  ) : (
    <div></div>
  );

export default class List extends Component {
  constructor(props) {
    super(props);
    this.state = {
      width: window.innerWidth,
      height: window.innerHeight,
    };
  }

  gridRef = null;
  rows = Math.ceil(this.props.fileList.length / this.props.columns);
  columns = this.props.columns;

  handleResize() {
    this.setState({ width: window.innerWidth, height: window.innerHeight });
  }
  componentDidMount() {
    this.gridRef.scrollToItem({
      columnIndex: this.props.cursor % this.columns,

      rowIndex: Math.floor(this.props.cursor / this.columns),
    });
    window.addEventListener("resize", this.handleResize.bind(this));
  }
  componentDidUpdate() {
    this.gridRef.scrollToItem({
      columnIndex: this.props.cursor % this.columns,

      rowIndex: Math.floor(this.props.cursor / this.columns),
    });
  }

  componentWillUnmount() {
    window.removeEventListener("resize", this.handleResize.bind(this));
  }

  shouldComponentUpdate(prevProps, prevState) {
    return (
      this.props.cursor !== prevProps.cursor ||
      this.props.items !== prevProps.items ||
      this.props.size !== prevProps.size ||
      this.props.shuffles !== prevProps.shuffles ||
      this.state.width !== prevState.width
    );
  }

  render() {
    const { fileList, useBucket, handleSelection, handleRightClick } = this.props;
    return (
      <div
        className="container"
        data-tid="container"
        onContextMenu={handleRightClick}
      >
        <Grid
          ref={(r) => {
            this.gridRef = r;
          }}
          columnCount={this.columns}
          columnWidth={this.state.width / this.props.columns}
          height={window.innerHeight}
          rowCount={this.rows}
          rowHeight={this.state.width / this.props.columns}
          overscanRowCount={0}
          width={this.state.width}
          itemData={{
            fileList,
            useBucket,
            handleSelection,
            columns: this.columns,
            size: this.props.size,
          }}
        >
          {Cell}
        </Grid>
      </div>
    );
  }
}
