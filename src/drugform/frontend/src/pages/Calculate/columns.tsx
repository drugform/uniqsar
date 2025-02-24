import { Typography } from "antd";
import { MolCalcResult } from "../../types/molecula";

export const columns = [
  {
    title: "Номер",
    dataIndex: "num",
    render: (first: any, record: MolCalcResult) => (
      <Typography.Text>{record.num}</Typography.Text>
    ),
    width: "80px",
  },
  {
    title: "Скор",
    dataIndex: "score",
    width: "80px",
    render: (first: any, record: MolCalcResult) => (
      <Typography.Text>{record.calcMol?.total.score}</Typography.Text>
    ),
  },
  {
    title: "SMILES",
    dataIndex: "smiles",
    render: (first: any, record: MolCalcResult) => (
      <Typography.Text
        ellipsis
        style={{
          fontSize: 10,
          lineHeight: "16px",
          display: "inline-block",
        }}
      >
        {record.mol.smiles}
      </Typography.Text>
    ),
  },
];
