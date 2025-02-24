import {
  Button,
  Flex,
  Input,
  PaginationProps,
  Table,
  Typography,
  Image,
  Empty,
  Upload,
  UploadProps,
  message,
  Form,
  Modal,
} from "antd";
import { ReactElement, useCallback, useMemo, useState } from "react";

import { MolCalc, Molecula, MolCalcResult } from "../../types/molecula";
import { columns } from "./columns";
import { isValidJson } from "../../utils/utils";
import requestService from "../../api/http";
import { endpoints } from "../../api/constants";
import "./styles.scss";
import { BaseResponse } from "../../types/common";
import { csvMaker } from "../../utils/csvMaker";
import { download } from "../../utils/fileDownload";
import { messageDuration } from "../../utils/constants";
import SmailesModal from "./SmilesModal";

interface TaskInput {
  taskParams: string;
}

function Calculate(): ReactElement {
  const [isLoading, setIsLoading] = useState(false);
  const [selectedMolecula, setSelectedMolecula] = useState<MolCalcResult>();
  const [moleculaList, setMoleculaList] = useState<Molecula[]>([]);
  const [calcResultMolList, setCalcResultMolList] = useState<MolCalc[]>([]);
  const [page, setPage] = useState(1);
  const [form] = Form.useForm<TaskInput>();
  const [modal, contextHolder] = Modal.useModal();
  const [openEncMolModal, setOpenEncMolModal] = useState(false);

  const handlePageChange: PaginationProps["onChange"] = (pageNumber) => {
    setPage(pageNumber);
  };

  const molCalcResultList = useMemo(
    () =>
      moleculaList.map((item, i) => {
        return {
          num: i + 1,
          mol: { ...item },
          calcMol: calcResultMolList[i]
            ? { ...calcResultMolList[i] }
            : undefined,
        } as MolCalcResult;
      }),
    [calcResultMolList, moleculaList]
  );

  const handleSubmit = useCallback(
    async (values: TaskInput) => {
      try {
        setIsLoading(true);
        const response = await requestService.post<BaseResponse<MolCalc[]>>(
          endpoints.calc,
          {
            taskParams: JSON.parse(values.taskParams),
            mols: moleculaList,
          }
        );
        if (!response.data.success) {
          modal.error({
            content: <pre>{response.data.message}</pre>,
          });
        } else {
          setCalcResultMolList(response.data.data || []);
          message.success("Произведен обсчет молекул", messageDuration);
        }
      } finally {
        setIsLoading(false);
      }
    },
    [modal, moleculaList]
  );

  const handleResetDataClick = useCallback(() => {
    setSelectedMolecula(undefined);
    setMoleculaList([]);
    setCalcResultMolList([]);
    form.resetFields();
  }, [form]);

  const props: UploadProps = {
    action: endpoints.encodeCsv,
    accept: ".csv",
    showUploadList: false,
    onChange(info) {
      if (info.file.status !== "uploading") {
        console.log(info.file, info.fileList);
      }
      if (info.file.status === "done") {
        const response: BaseResponse<Molecula[]> = info.file.response;
        if (response.success) {
          setMoleculaList(response.data || []);
          message.success("Файл успешно обработан", messageDuration);
        } else {
          modal.error({
            content: <pre>{response.message || "Ошибка обработки файла"}</pre>,
          });
        }
      } else if (info.file.status === "error") {
        modal.error({
          content: `ошибка загрузки файла ${info.file.name}`,
        });
      }
    },
  };

  const getReport = useCallback(() => {
    const list = molCalcResultList.map((item) => ({
      id: item.num,
      score: item.calcMol?.total.score,
      smiles: item.mol.smiles,
      result: JSON.stringify(item.calcMol),
    }));
    const csvData = list.map((item) => csvMaker(item));
    const headers = "id,score,smiles,result";
    const result = [headers, ...csvData].join("\n");
    download(result);
  }, [molCalcResultList]);

  const addMolClick = useCallback(() => {
    setOpenEncMolModal(true);
  }, []);

  const onCloseEncMolClick = useCallback(() => {
    setOpenEncMolModal(false);
  }, []);

  const onCalcMol = useCallback((mol: Molecula) => {
    setMoleculaList([mol]);
    setCalcResultMolList([]);
    setSelectedMolecula(undefined);
  }, []);

  return (
    <Flex className="calculate" gap={24}>
      <Flex vertical gap={16} className="calculate__taskInfo">
        <div className="calculate__taskInfo__upload">
          <Upload {...props}>
            <Button type="primary">Загрузить молекулы</Button>
          </Upload>
        </div>
        <div className="calculate__taskDescription">
          <Form
            form={form}
            onFinish={handleSubmit}
            className="w-100"
            initialValues={{
              taskParams: "",
            }}
          >
            <Form.Item
              name="taskParams"
              rules={[
                {
                  required: true,
                  message: "Задача не сформулирована",
                },
                {
                  message: "JSON не валидный",
                  validator: (_, value) => {
                    if (isValidJson(value)) {
                      return Promise.resolve();
                    } else {
                      return Promise.reject("JSON не валидный");
                    }
                  },
                },
              ]}
            >
              <Input.TextArea
                placeholder="Формулировка задачи"
                showCount
                style={{ height: 300, resize: "none", width: "100%" }}
              />
            </Form.Item>
            <Form.Item style={{ margin: 0 }}>
              <Flex justify="center" align="center">
                <Button
                  type="primary"
                  className="w-100"
                  htmlType="submit"
                  loading={isLoading}
                >
                  Запустить расчет
                </Button>
              </Flex>
            </Form.Item>
          </Form>
        </div>
      </Flex>
      <Flex vertical gap={16} className="calculate__taskResult">
        <Flex gap={16} className="calculate__buttons">
          <Button type="primary" onClick={handleResetDataClick}>
            Сброс результатов
          </Button>
          <Button
            type="primary"
            onClick={getReport}
            disabled={!molCalcResultList.length}
          >
            Выгрузка результатов
          </Button>
          <Button type="primary" onClick={addMolClick}>
            Добавить молекулу
          </Button>
        </Flex>
        <Table
          dataSource={molCalcResultList}
          columns={columns}
          onRow={(record) => ({
            onClick: () => setSelectedMolecula(record),
          })}
          rowKey="id"
          rowClassName={(record) => {
            const classList = [];
            if (record.num === selectedMolecula?.num) {
              classList.push("active");
            }
            return classList.join(" ");
          }}
          pagination={{
            position: ["bottomCenter"],
            pageSize: 15,
            total: moleculaList.length,
            onChange: handlePageChange,
            hideOnSinglePage: true,
            showSizeChanger: false,
            current: page,
          }}
        />
      </Flex>
      {selectedMolecula && (
        <Flex vertical gap={16} className="calculate__moleculaView">
          <Typography.Text className="calculate__moleculaView__title">
            Просмотр молекулы
          </Typography.Text>
          <Typography.Text className="calculate__moleculaView__label">
            SMILES
          </Typography.Text>
          <Typography.Text className="calculate__moleculaView__value">
            {selectedMolecula.mol.smiles}
          </Typography.Text>
          <Image src={selectedMolecula.mol.imgurl} width={400} height={300} />
          <Flex gap={8} vertical>
            <Typography.Text className="calculate__moleculaView__label">
              Детали расчета
            </Typography.Text>
            <Typography.Text className="calculate__moleculaView__value">
              <pre>{JSON.stringify(selectedMolecula.calcMol, null, 2)}</pre>
            </Typography.Text>
          </Flex>
        </Flex>
      )}
      {!selectedMolecula && (
        <Flex vertical gap={16} className="calculate__moleculaView">
          <Empty description="Для просмотра выберите молекулу" />
        </Flex>
      )}
      {contextHolder}
      <SmailesModal
        open={openEncMolModal}
        onClose={onCloseEncMolClick}
        onSuccess={onCalcMol}
      />
    </Flex>
  );
}

export default Calculate;
