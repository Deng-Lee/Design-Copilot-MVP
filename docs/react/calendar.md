
```jsx
import React, { useState } from 'react';
import { Calendar } from 'antd';
import dayjs from 'dayjs';
import 'dayjs/locale/zh-cn';
import calendar from 'dayjs/plugin/calendar';

// 配置 dayjs 使用中文和农历插件
dayjs.locale('zh-cn');
dayjs.extend(calendar);

const LunarCalendar = () => {
  const [value, setValue] = useState(() => dayjs());

  // 自定义日期单元格渲染
  const dateCellRender = (current) => {
    // 获取农历日期
    const lunarDate = current.calendar();
    
    return (
      <div className="ant-picker-cell-inner">
        <div className="ant-picker-calendar-date-value">
          {current.date()}
        </div>
        <div className="ant-picker-calendar-date-content">
          <div style={{ fontSize: '12px', color: '#8c8c8c' }}>
            {lunarDate}
          </div>
        </div>
      </div>
    );
  };

  // 自定义月份单元格渲染
  const monthCellRender = (current) => {
    // 获取农历月份
    const lunarMonth = current.calendar();
    
    return (
      <div className="ant-picker-cell-inner">
        <div className="ant-picker-calendar-date-value">
          {current.month() + 1}月
        </div>
        <div className="ant-picker-calendar-date-content">
          <div style={{ fontSize: '12px', color: '#8c8c8c' }}>
            {lunarMonth}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ padding: 24, maxWidth: 800, margin: '0 auto' }}>
      <h2>农历日历示例</h2>
      <Calendar
        value={value}
        onChange={setValue}
        dateCellRender={dateCellRender}
        monthCellRender={monthCellRender}
        headerRender={({ value: currentValue, onChange }) => {
          const year = currentValue.year();
          const month = currentValue.month();
          
          return (
            <div style={{ padding: 8, textAlign: 'center' }}>
              <button
                onClick={() => onChange(currentValue.subtract(1, 'month'))}
                style={{ marginRight: 8 }}
              >
                上个月
              </button>
              <span style={{ margin: '0 16px' }}>
                {year}年 {month + 1}月
              </span>
              <button
                onClick={() => onChange(currentValue.add(1, 'month'))}
                style={{ marginLeft: 8 }}
              >
                下个月
              </button>
            </div>
          );
        }}
      />
    </div>
  );
};

export default LunarCalendar;
```

**安装依赖：**

```bash
npm install antd dayjs
# 或
yarn add antd dayjs
```
