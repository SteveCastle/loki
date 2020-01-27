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
        overflow: "hidden"
      }}
    >
      <ListItem
        className="listImage"
        handleClick={() =>
          data.handleClick(rowIndex * data.columns + columnIndex)
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
      height: window.innerHeight
    };
  }

  gridRef = null;
  rows = this.props.tall
    ? Math.ceil(this.props.fileList.length / 3)
    : Math.ceil(Math.sqrt(this.props.fileList.length));
  columns = this.props.tall
    ? 3
    : Math.ceil(Math.sqrt(this.props.fileList.length));

  handleResize() {
    console.log("resizing to:", window.innerWidth);
    this.setState({ width: window.innerWidth, height: window.innerHeight });
  }
  componentDidMount() {
    this.gridRef.scrollToItem({
      columnIndex: this.props.cursor % this.columns,

      rowIndex: Math.floor(this.props.cursor / this.columns)
    });
    window.addEventListener("resize", this.handleResize.bind(this));
  }
  componentDidUpdate() {
    this.gridRef.scrollToItem({
      columnIndex: this.props.cursor % this.columns,

      rowIndex: Math.floor(this.props.cursor / this.columns)
    });
  }

  componentWillUnmount() {
    window.removeEventListener("resize", this.handleResize.bind(this));
  }

  render() {
    const { fileList, handleClick, filter } = this.props;
    return (
      <div className="container" data-tid="container">
        <HotKeyController handleKeyPress={this.props.handleKeyPress} />
        <Grid
          ref={r => {
            this.gridRef = r;
          }}
          columnCount={this.columns}
          columnWidth={this.state.width / 3}
          height={window.innerHeight}
          rowCount={this.rows}
          rowHeight={this.state.width / 3}
          overscanRowCount={0}
          width={this.state.width}
          itemData={{ fileList, handleClick, columns: this.columns }}
        >
          {Cell}
        </Grid>
      </div>
    );
  }
}
