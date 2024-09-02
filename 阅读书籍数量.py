import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv('./book_read.csv')
print(df.head())
print(df.isna().sum())

x = df[['HoursSpentReading']]
y = df['BooksRead']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(f'线性回归 mse:{mse} , r2:{r2 * 100:.2f}%')

true_values = y_test.tolist()
predicted_values = y_pred.tolist()

line_chart = (
    Line(init_opts=opts.InitOpts(width="1000px", height="600px"))
    .add_xaxis(list(range(len(true_values))))
    .add_yaxis("True Values", true_values, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("Predicted Values", predicted_values, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="每周阅读时间预测每月阅读书籍数量对比"),
        xaxis_opts=opts.AxisOpts(name="样本索引"),
        yaxis_opts=opts.AxisOpts(name="每月阅读书籍数量"),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        toolbox_opts=opts.ToolboxOpts(),
        datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")]
    )
)

line_chart.render('线性回归预测阅读时间.html')

# https://www.kaggle.com/datasets/trushildhokiya/number-of-books-read-simple-linear-regression
