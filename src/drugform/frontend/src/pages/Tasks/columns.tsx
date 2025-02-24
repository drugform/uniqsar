import { Typography } from "antd";
import { TaskResult } from "../../types/task";

export const columns = [
  {
    title: "Номер",
    dataIndex: "id",
    render: (first: any, record: TaskResult) => record.id,
    width: "8%",
  },
  {
    title: "Скор",
    dataIndex: "score",
    width: "8%",
    render: (first: any, record: TaskResult) => record.score,
  },
  {
    title: "SMILES",
    dataIndex: "smiles",
    width: "86%",
    render: (first: any, record: TaskResult) => (
      <Typography.Text
        ellipsis
        style={{
          fontSize: 10,
          lineHeight: "16px",
          display: "inline-block",
        }}
      >
        {record.mol?.smiles}
      </Typography.Text>
    ),
  },
];
