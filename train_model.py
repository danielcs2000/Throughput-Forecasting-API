import pandas as pd
import numpy as np
import datetime
import tensorflow as tf


df = pd.read_csv('data/data_3.csv', names=['timestep', 'value'])
df = df.set_index('timestep')

MAX_SAMPLE_SIZE=len(df)-500


class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None, time_step=0):
    # Store the raw data.

    self.time_step = time_step
    self.label_columns = label_columns

    if self.time_step:
      self.train_df = self.unrolling(train_df)
      self.val_df = self.unrolling(val_df)
      self.test_df = self.unrolling(test_df)
    else:
      self.train_df = train_df
      self.val_df = val_df
      self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels
  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=200,)


    ds = ds.map(self.split_window)

    return ds
  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  def unrolling(self, df):
    size_step = (len(df) + 1 - self.time_step) // self.time_step
    df_x = df.index.values
    X = []
    Y_s = {}
    index_name = df.index.name

    for i in range(size_step - 1):
      x = df_x[i * self.time_step:(i + 1) * self.time_step]
      for column_name in self.label_columns:
        if column_name not in Y_s:
          Y_s[column_name] = []
        y = df[column_name][i * self.time_step:(i + 1) * self.time_step]
        Y_s[column_name].extend(y)  # Add elements of one list to another list
      X.extend(x.tolist())  # tolist from normal list to nested list
    X.extend((df_x[(i + 1) * self.time_step:]).tolist())
    for column_name in self.label_columns:
        Y_s[column_name].extend((df[column_name][(i + 1) * self.time_step:]).tolist())
    df_dict = Y_s
    df_dict['index'] = X
    new_df = pd.DataFrame.from_dict(df_dict)
    new_df = new_df.set_index('index')
    return new_df

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result
  

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min',
                                                    verbose=1)

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0006),
                metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])

  history = model.fit(window.train, epochs=2000,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history



rnn_unit = 20  # hidden layer units
input_size = 1  # 输入层维度
output_size = 1  # 输出层维度
lr = 0.0006  # 学习率
# batch_size = 60   # 每批次数据个数
time_step = 20  # Unrolling  LSTM的展开
#TRAIN_BATCH_SIZE=200



headers = ['Station', 'Part', 'ME', 'MAE', 'RMSE', 'MRE']

def lstm_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(rnn_unit, input_shape=(time_step, input_size), return_sequences=False,
                             kernel_initializer=tf.keras.initializers.RandomNormal(),
                    bias_initializer=tf.keras.initializers.RandomNormal(),
                    name='LSTM_layer'),
        tf.keras.layers.Dense(1,
                    kernel_initializer=tf.keras.initializers.RandomNormal(),
                    bias_initializer=tf.keras.initializers.RandomNormal(),
                    name='bias_output_layer')
    ])
    return model


n = len(df)
train_df = df[0:int(n*0.9)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]
num_features = df.shape[1]

# build train dataset
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std

# build test dataset
test_mean = test_df.mean()
test_std = test_df.std()
test_df = (test_df - test_mean) / test_std

# build val dataset
val_mean = val_df.mean()
val_std = val_df.std()
val_df = (val_df - val_mean) / val_std



single_step_window = WindowGenerator(input_width=time_step, label_width=input_size, shift=1,
                                     train_df=train_df, val_df=val_df, test_df=test_df,
                                     label_columns=['value'], time_step=0)

print('Input shape:', single_step_window)
single_step_window.train

model = lstm_model()
history = compile_and_fit(model, single_step_window, patience=10)
print(model.summary())
#_, val_mae, val_rmse = model.evaluate(single_step_window.val)
_, test_mae, test_rmse = model.evaluate(single_step_window.test, verbose=1)
print(test_mae, test_rmse)


model.save('model/LSTM-ST-Model.keras')