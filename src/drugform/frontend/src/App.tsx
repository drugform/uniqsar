import { Button, Flex } from "antd";

import { NavLink, Route, Routes } from "react-router-dom";
import Tasks from "./pages/Tasks";
import CreateTask from "./pages/CreateTask";
import Calculate from "./pages/Calculate";
import { AppRoutes } from "./constants";

import "./App.scss";

function App() {
  return (
    <Flex vertical className="moleculaRoot">
      <div className="content-wrapper">
        <Flex className="moleculaRoot__header " gap={16}>
          <NavLink to={AppRoutes.home}>
            <Button type="link" className="moleculaRoot__header__menuItem">
              Все задачи
            </Button>
          </NavLink>
          <NavLink to={AppRoutes.createTask}>
            <Button type="link" className="moleculaRoot__header__menuItem">
              Новая задача
            </Button>
          </NavLink>
          <NavLink to={AppRoutes.calc}>
            <Button type="link" className="moleculaRoot__header__menuItem">
              Расчет свойств
            </Button>
          </NavLink>
        </Flex>
      </div>
      <div className="content-wrapper">
        <Flex vertical className="moleculaRoot__content">
          <Routes>
            <Route path={AppRoutes.home} element={<Tasks />} />
            <Route path={AppRoutes.createTask} element={<CreateTask />} />
            <Route path={AppRoutes.calc} element={<Calculate />} />
            <Route path="*" element={<Tasks />} />
          </Routes>
        </Flex>
      </div>
    </Flex>
  );
}

export default App;
