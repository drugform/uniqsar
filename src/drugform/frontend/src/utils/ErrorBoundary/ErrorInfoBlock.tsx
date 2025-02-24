import { Result, Button, Collapse, Spin } from "antd";
import { Suspense } from "react";

interface ErrorInfoBlockProps {
  error: Error | null;
}

function ErrorInfoBlock({ error }: ErrorInfoBlockProps) {
  return (
    <Suspense fallback={<Spin spinning />}>
      <Result
        status="warning"
        title="Что-то пошло не так."
        extra={
          <Button
            type="primary"
            onClick={() => {
              window.location.href = "/";
            }}
          >
            На главную
          </Button>
        }
      />
      <Collapse
        style={{
          margin: "16px",
        }}
        items={[
          {
            key: "1",
            label: "Error",
            children: (
              <>
                <h4>error.message</h4>
                <pre>{error?.message}</pre>
                <h4>error.stack</h4>
                <pre>{error?.stack}</pre>
              </>
            ),
          },
        ]}
      />
    </Suspense>
  );
}

export default ErrorInfoBlock;
