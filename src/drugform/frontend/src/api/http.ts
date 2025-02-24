import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from "axios";

export const API_CONTEXT = "/api";

const headers: Readonly<Record<string, string | boolean>> = {
  Accept: "application/json",
  "Content-Type": "application/json; charset=utf-8",
  "Access-Control-Allow-Credentials": true,
  "X-Requested-With": "XMLHttpRequest",
  Authorization: "Bearer dGVzdDp0ZXN0",
};

interface BaseError {
  status: number;
  config: any;
}

class Http {
  private static httpInstance: Http;

  private http: AxiosInstance;

  constructor() {
    this.http = axios.create({
      baseURL: API_CONTEXT,
      headers,
      withCredentials: true,
    });

    this.http.interceptors.response.use(
      (response) => response,
      (error) => {
        const { response } = error;
        return this.handleError(response);
      }
    );
  }

  static getInstance(): Http {
    if (!this.httpInstance) {
      this.httpInstance = new this();
    }

    return this.httpInstance;
  }

  private handleError = async (error: BaseError) => {
    const { status } = error;
    switch (status) {
      case 500: {
        break;
      }
      default:
        break;
    }

    return Promise.reject(error);
  };

  get<T = any, R = AxiosResponse<T>>(
    url: string,
    config?: AxiosRequestConfig
  ): Promise<R> {
    return this.http.get<T, R>(url, config);
  }

  post<T = any, R = AxiosResponse<T>, D = any>(
    url: string,
    data?: D,
    config?: AxiosRequestConfig<D>
  ): Promise<R> {
    return this.http.post<T, R>(url, data, config);
  }

  put<T = any, R = AxiosResponse<T>, D = any>(
    url: string,
    data?: D,
    config?: AxiosRequestConfig<D>
  ): Promise<R> {
    return this.http.put<T, R>(url, data, config);
  }

  delete<T = any, R = AxiosResponse<T>>(url: string): Promise<R> {
    return this.http.delete<T, R>(url);
  }
}

const requestService = Http.getInstance();
export default requestService;
