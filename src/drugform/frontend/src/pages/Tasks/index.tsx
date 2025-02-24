import {
  Button,
  Collapse,
  CollapseProps,
  Empty,
  Flex,
  Image,
  message,
  Modal,
  PaginationProps,
  Skeleton,
  Table,
  Typography,
} from "antd";
import { ReactElement, useCallback, useEffect, useState } from "react";
import { Task, TaskResult } from "../../types/task";
import { columns } from "./columns";
import clsx from "clsx";
import requestService from "../../api/http";
import { BaseResponse } from "../../types/common";
import { endpoints } from "../../api/constants";
import { csvMaker } from "../../utils/csvMaker";
import { download } from "../../utils/fileDownload";
import { messageDuration } from "../../utils/constants";
import "./styles.scss";
import { getDate } from "../../utils/utils";
import { useSearchParams } from "react-router-dom";

function Tasks(): ReactElement {
  const [taskList, setTaskList] = useState<Task[]>([]);
  const [selectedTask, setSelectedTask] = useState<Task>();
  const [taskResultList, setTaskResultList] = useState<TaskResult[]>([]);
  const [selectedTaskResult, setSelectedTaskResult] = useState<TaskResult>();
  const [page, setPage] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingResult, setIsLoadingResult] = useState(false);
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(false);
  const [modal, contextHolder] = Modal.useModal();
  const [searchParams] = useSearchParams();

  useEffect(() => {
    const defaultTaskId = searchParams.get("id");
    if (!selectedTask && taskList.length > 0) {
      if (defaultTaskId) {
        const foundTask = taskList.find(
          (item) => item.taskId === defaultTaskId
        );
        if (foundTask) {
          setSelectedTask(foundTask);
        } else {
          setSelectedTask(taskList[0]);
        }
      } else {
        setSelectedTask(taskList[0]);
      }
    }
  }, [searchParams, selectedTask, taskList]);

  const load = useCallback(async () => {
    try {
      setIsLoading(true);
      const response = await requestService.get<BaseResponse<string[]>>(
        endpoints.generateInfo
      );
      if (!response.data.success) {
        modal.error({
          content: <pre>{response.data.message}</pre>,
        });
      } else {
        const requests: Promise<any>[] = [];
        const taskIds = response.data.data || [];
        taskIds.forEach((id) => {
          requests.push(
            requestService.post<BaseResponse<Task>>(
              endpoints.generateInfoByTask(id)
            )
          );
        });
        const responses = await Promise.all(requests);
        if (responses.length === taskIds.length) {
          setTaskList(
            responses.map((res) => ({
              ...res.data.data,
              generatorParams: JSON.stringify(
                res.data.data.generatorParams,
                null,
                2
              ),
              taskParams: JSON.stringify(res.data.data.taskParams, null, 2),
              taskInfo: JSON.stringify(res.data.data.taskInfo, null, 2),
              name: res.data.data.taskInfo?.name || "",
            }))
          );
        } else {
          message.error("Ошибка получения данных", messageDuration);
        }
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  }, [modal]);

  useEffect(() => {
    load();
  }, [load]);

  const loadTaskResult = useCallback(
    async (taskId: string) => {
      try {
        setIsLoadingResult(true);
        setTaskResultList([]);
        const response = await requestService.get<BaseResponse<TaskResult[]>>(
          endpoints.generateTaskResult(taskId, 100)
        );
        if (response.data.success) {
          const list = response.data.data || [];
          setTaskResultList(list.map((item, i) => ({ ...item, id: i + 1 })));
        } else {
          modal.error({
            content: (
              <pre>
                {response.data.message ||
                  `Ошибка получения данных по задаче ${taskId}`}
              </pre>
            ),
          });
        }
      } catch (e) {
        console.error(e);
      } finally {
        setIsLoadingResult(false);
      }
    },
    [modal]
  );

  const generateTaskResult = useCallback(() => {
    if (selectedTask) {
      loadTaskResult(selectedTask.taskId);
    }
  }, [loadTaskResult, selectedTask]);

  useEffect(() => {
    if (selectedTask) {
      loadTaskResult(selectedTask.taskId);
    }
  }, [loadTaskResult, selectedTask]);

  const handlePageChange: PaginationProps["onChange"] = (pageNumber) => {
    setPage(pageNumber);
  };

  const onStopGenerateTaskResult = useCallback(async () => {
    if (!selectedTask) {
      return;
    }
    try {
      const response = await requestService.post<BaseResponse<TaskResult[]>>(
        endpoints.generateTaskStop(selectedTask.taskId)
      );
      if (response.data.success) {
        message.success("Генерация остановлена", messageDuration);
      } else {
        modal.error({
          content: (
            <pre>{response.data.message || "Ошибка остановки расчетов"}</pre>
          ),
        });
      }
    } catch (e) {
      console.error(e);
    }
  }, [modal, selectedTask]);

  const getReport = useCallback(() => {
    const list = taskResultList.map((item) => ({
      id: item.id,
      smiles: JSON.stringify(item.mol?.smiles),
      score: item.score,
      value: JSON.stringify(item.value),
    }));
    const csvData = list.map((item) => csvMaker(item));
    const headers = "id,smiles,score,value";
    const result = [headers, ...csvData].join("\n");
    download(result);
  }, [taskResultList]);

  const loadMetrics = useCallback(async () => {
    if (!selectedTask) {
      return;
    }
    try {
      setIsLoadingMetrics(true);
      const response = await requestService.get<Blob>(
        endpoints.generateTaskMetrics(selectedTask.taskId),
        {
          responseType: "blob",
        }
      );
      if (response.data) {
        const blobUrl = URL.createObjectURL(response.data);
        window.open(blobUrl, "_blank");
      } else {
        message.error("Ошибка получения изображения");
      }
    } catch (e) {
      console.error(e);
      modal.error({
        content: <pre>Ошибка генерации метрики</pre>,
      });
    } finally {
      setIsLoadingMetrics(false);
    }
  }, [modal, selectedTask]);

  if (!isLoading && taskList.length === 0) {
    return <Empty description="Список задач пуст" />;
  }

  const items: CollapseProps["items"] = [
    {
      key: "1",
      label: "Параметры задачи",
      children: (
        <Flex gap={24} align="flex-start">
          <Flex vertical gap={4} style={{ flex: "0.5" }}>
            <Typography.Text className="tasks__description">
              Описание задачи: <br /> <pre>{selectedTask?.taskInfo}</pre>
            </Typography.Text>
            <Typography.Text className="tasks__description">
              Параметры задачи: <br />
              <pre>{selectedTask?.taskParams}</pre>
            </Typography.Text>
          </Flex>
          <Flex vertical gap={4} style={{ flex: "0.5" }}>
            <Typography.Text className="tasks__description">
              Параметры генерации: <br />
              <pre>{selectedTask?.generatorParams}</pre>
            </Typography.Text>
          </Flex>
        </Flex>
      ),
    },
  ];

  return (
    <Skeleton loading={isLoading} active>
      <Flex className="tasks" gap={32}>
        <Flex vertical gap={8} className="tasks__list">
          {taskList.map((taskItem) => (
            <Flex
              key={taskItem.taskId}
              vertical
              gap={4}
              className={clsx("tasks__item", {
                active: taskItem.taskId === selectedTask?.taskId,
              })}
              onClick={() => setSelectedTask(taskItem)}
            >
              <Typography.Text className="tasks__name">
                {taskItem.taskId}
              </Typography.Text>
              <Typography.Text className="tasks__name">
                {taskItem.name}
              </Typography.Text>
              <Typography.Text className="tasks__date">
                <b>{getDate(taskItem.startTime)}</b>
              </Typography.Text>
            </Flex>
          ))}
        </Flex>
        {selectedTask && (
          <Flex gap={24} vertical className="tasks__viewSelected">
            <Flex gap={24} vertical className="tasks__info">
              <Flex gap={16} justify="space-between">
                <Flex vertical gap={8}>
                  <Typography.Text className="tasks__description">
                    Имя задачи: {selectedTask.name}
                  </Typography.Text>
                  <Typography.Text className="tasks__date">
                    Время начала: <b>{getDate(selectedTask.startTime)}</b>
                  </Typography.Text>
                  <Typography.Text className="tasks__date">
                    Время завершения: <b>{getDate(selectedTask.endTime)}</b>
                  </Typography.Text>
                </Flex>
                <Flex
                  gap={8}
                  wrap="wrap"
                  justify="flex-end"
                  style={{ maxWidth: 260 }}
                >
                  <Button
                    style={{ width: 120 }}
                    type="primary"
                    onClick={generateTaskResult}
                  >
                    Обновить
                  </Button>
                  <Button
                    style={{ width: 120 }}
                    type="primary"
                    onClick={onStopGenerateTaskResult}
                  >
                    Остановить
                  </Button>
                  <Button
                    style={{ width: 120 }}
                    type="primary"
                    onClick={getReport}
                    disabled={!taskResultList.length}
                  >
                    Выгрузить
                  </Button>
                  <Button
                    style={{ width: 120 }}
                    type="primary"
                    onClick={loadMetrics}
                    disabled={!selectedTask}
                    loading={isLoadingMetrics}
                  >
                    Метрики
                  </Button>
                </Flex>
              </Flex>
              <Collapse ghost items={items} />
            </Flex>
            <Table
              loading={isLoadingResult}
              dataSource={taskResultList}
              columns={columns}
              onRow={(record) => ({
                onClick: () => setSelectedTaskResult(record),
              })}
              rowKey="id"
              rowClassName={(record) => {
                const classList = [];
                if (record.id === selectedTaskResult?.id) {
                  classList.push("active");
                }
                return classList.join(" ");
              }}
              pagination={{
                position: ["bottomCenter"],
                pageSize: 15,
                total: taskResultList.length,
                onChange: handlePageChange,
                hideOnSinglePage: true,
                showSizeChanger: false,
                current: page,
              }}
            />
          </Flex>
        )}
        {!selectedTask && <Empty description="Для просмотра выберите задачу" />}

        {selectedTaskResult && (
          <Flex vertical gap={16} className="tasks__item__result">
            <Typography.Text className="tasks__item__result__title">
              Просмотр молекулы
            </Typography.Text>
            <Typography.Text className="tasks__item__result__label">
              SMILES
            </Typography.Text>
            <Typography.Text className="tasks__item__result__description">
              {selectedTaskResult.mol?.smiles}
            </Typography.Text>
            <Image
              src={selectedTaskResult.mol?.imgurl}
              width={400}
              height={300}
            />
            <Flex gap={8} vertical>
              <Typography.Text className="tasks__item__result__label">
                Детали расчета
              </Typography.Text>
              <Typography.Text className="tasks__item__result__description">
                <pre>{JSON.stringify(selectedTaskResult.value, null, 2)}</pre>
              </Typography.Text>
            </Flex>
          </Flex>
        )}
        {!selectedTaskResult && (
          <Flex vertical gap={16} className="tasks__item__result">
            <Empty description="Для просмотра молекулы выберите результат по задаче" />
          </Flex>
        )}
      </Flex>
      {contextHolder}
    </Skeleton>
  );
}

export default Tasks;
