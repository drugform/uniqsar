export interface BaseError {
  message?: string;
  error?: string;
}

export interface BaseResponse<T> extends BaseError {
  success: boolean;
  data?: T;
}
