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
        handleClick={() =>
          data.handleClick(rowIndex * data.columns + columnIndex)
        }
        handleRightClick={data.setPath}
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
  rows = this.props.tall
    ? Math.ceil(this.props.fileList.length / 3)
    : Math.ceil(Math.sqrt(this.props.fileList.length));
  columns = this.props.tall
    ? 3
    : Math.ceil(Math.sqrt(this.props.fileList.length));

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
    const { fileList, handleClick, setPath } = this.props;
    return (
      <div className="container" data-tid="container">
        <Grid
          ref={(r) => {
            this.gridRef = r;
          }}
          columnCount={this.columns}
          columnWidth={this.state.width / 3}
          height={window.innerHeight}
          rowCount={this.rows}
          rowHeight={this.state.width / 3}
          overscanRowCount={0}
          width={this.state.width}
          itemData={{
            fileList,
            handleClick,
            setPath,
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
