import { Button, Flex, Form, Input, message, Modal } from "antd";
import { ReactElement, useCallback, useState } from "react";

import requestService from "../../api/http";
import { endpoints } from "../../api/constants";
import { CreateTaskResponse } from "../../types/task";
import { isValidJson } from "../../utils/utils";
import { BaseResponse } from "../../types/common";
import { messageDuration } from "../../utils/constants";

import "./styles.scss";
import { useNavigate } from "react-router-dom";
import { AppRoutes } from "../../constants";

interface CreateTaskInput {
  taskParams: string;
  taskInfo: string;
  generatorParams: string;
}

function CreateTask(): ReactElement {
  const [isLoading, setIsLoading] = useState(false);
  const [form] = Form.useForm<CreateTaskInput>();
  const [modal, contextHolder] = Modal.useModal();
  const navigate = useNavigate();

  const handleSubmit = useCallback(
    async (values: CreateTaskInput) => {
      try {
        setIsLoading(true);
        const response = await requestService.post<
          BaseResponse<CreateTaskResponse>
        >(endpoints.generate, {
          taskInfo: JSON.parse(values.taskInfo),
          taskParams: JSON.parse(values.taskParams),
          generatorParams: JSON.parse(values.generatorParams),
        });
        if (!response.data.success) {
          modal.error({
            content: <pre>{response.data.message}</pre>,
          });
        } else {
          message.success(
            `Создана задача с taskId=${response.data?.data?.taskId}. Редирект на список задач`,
            messageDuration
          );
          form.resetFields();
          setTimeout(() => {
            navigate(`${AppRoutes.home}?id=${response.data?.data?.taskId}`);
          }, 2000);
        }
      } finally {
        setIsLoading(false);
      }
    },
    [form, modal, navigate]
  );

  return (
    <Form
      form={form}
      onFinish={handleSubmit}
      className="w-100"
      initialValues={{
        generatorParams: "",
        taskParams: "",
        taskInfo: "",
      }}
    >
      <Flex gap={24} className="w-100">
        <Form.Item
          style={{ flex: 0.3 }}
          name="taskInfo"
          rules={[
            {
              required: true,
              message: "Не указана информаия о задаче",
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
            placeholder="taskInfo"
            showCount
            style={{ height: 400, resize: "none", width: "100%" }}
            disabled={isLoading}
          />
        </Form.Item>

        <Form.Item
          name="taskParams"
          style={{ flex: 0.5 }}
          rules={[
            {
              required: true,
              message: "Цель не сформулирована",
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
            placeholder="taskParams"
            showCount
            style={{ height: 400, resize: "none", width: "100%" }}
            disabled={isLoading}
          />
        </Form.Item>

        <Form.Item
          name="generatorParams"
          style={{ flex: 0.5 }}
          rules={[
            {
              required: true,
              message: "Не указаны параметры генератора",
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
            placeholder="generatorParams"
            showCount
            style={{ height: 400, resize: "none", width: "100%" }}
            disabled={isLoading}
          />
        </Form.Item>
      </Flex>
      <Form.Item style={{ margin: 0 }}>
        <Flex justify="center" align="center">
          <Button type="primary" htmlType="submit" loading={isLoading}>
            Запустить задачу
          </Button>
        </Flex>
      </Form.Item>
      {contextHolder}
    </Form>
  );
}

export default CreateTask;
