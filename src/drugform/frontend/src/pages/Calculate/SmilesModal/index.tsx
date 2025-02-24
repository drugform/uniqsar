import { Button, Flex, Form, Input, message, Modal } from "antd";
import { ReactElement, useCallback, useState } from "react";
import requestService from "../../../api/http";
import { endpoints } from "../../../api/constants";
import { BaseResponse } from "../../../types/common";
import { Molecula } from "../../../types/molecula";
import { messageDuration } from "../../../utils/constants";

interface SmailesModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess: (mol: Molecula) => void;
}

interface MoleculaInput {
  smiles: string;
}

function SmailesModal({
  open,
  onSuccess,
  onClose,
}: SmailesModalProps): ReactElement {
  const [isLoading, setIsLoading] = useState(false);
  const [form] = Form.useForm<MoleculaInput>();
  const [modal, contextHolder] = Modal.useModal();

  const handleSubmit = useCallback(
    async (values: MoleculaInput) => {
      try {
        setIsLoading(true);
        const response = await requestService.post<BaseResponse<Molecula>>(
          endpoints.encodeMol,
          {
            ...values,
          }
        );
        if (!response.data.success) {
          modal.error({
            content: <pre>{response.data.message}</pre>,
          });
        } else {
          if (response.data.data) {
            onSuccess(response.data.data);
            message.success("Произведен обсчет молекулы", messageDuration);
            setTimeout(onClose, 2000);
          }
        }
      } finally {
        setIsLoading(false);
      }
    },
    [modal, onClose, onSuccess]
  );

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      mask={false}
      maskClosable={false}
      closeIcon
      width="600px"
    >
      <div style={{ padding: "32px 24px" }}>
        <Form
          form={form}
          onFinish={handleSubmit}
          className="w-100"
          initialValues={{
            taskParams: "",
          }}
        >
          <Form.Item
            name="smiles"
            rules={[
              {
                required: true,
                message: "Укажите smiles",
              },
            ]}
          >
            <Input.TextArea
              placeholder="Smiles"
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
                Добавить молекулу
              </Button>
            </Flex>
          </Form.Item>
        </Form>
        {contextHolder}
      </div>
    </Modal>
  );
}

export default SmailesModal;
