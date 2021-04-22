import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import time

np.random.seed(0)

# section rerun model with different hparams
st.title('Hyperparameters')
#with st.spinner('Wait for it...'):
#    time.sleep(5)
#st.success('Done!')

C = st.slider('reg', 0.001, 5., 1., step=0.001)

# utils
def exp_smooth(x, a):
    x_smooth = np.zeros_like(x)
    x_smooth[0] = x[0]

    for i in range(1, len(x)):
        x_smooth[i] = (1-a)*x[i-1] + a*x_smooth[i-1]
    return x_smooth

# trial data ##################################################################
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
y[y>1] = 1

X_train, X_test, y_train, y_test = train_test_split(X[:, 1:2], y, test_size=0.33)

clf = LogisticRegression(random_state=0, C=C).fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]

train_loss = np.exp(-np.arange(0, 5, 0.01)) + 0.1*np.random.normal(size=(500,))

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
X_embed = pca.transform(X)
###############################################################################

st.title('Results Summary')
st.header('Loss')

a = st.slider('Exponential smoothing factor', 0., 0.999, 0., step=0.001)

data = pd.DataFrame({
  'iteration': np.arange(len(train_loss)),
  'loss': train_loss,
  'smooth loss': exp_smooth(train_loss, a)})

chart_loss = alt.Chart(data).mark_line(opacity=0.5).encode(
    x='iteration',
    y='loss',
    color=alt.value('blue'))
chart_smooth_loss = alt.Chart(data).mark_line().encode(
    x='iteration',
    y='smooth loss',
    color=alt.value('blue'))

st.altair_chart(chart_loss+chart_smooth_loss, use_container_width=True)

# https://altair-viz.github.io/gallery/interactive_legend.html


# matrix plot
# smooth loss
from sklearn.metrics import f1_score, log_loss, accuracy_score, roc_curve


def get_max_score(y, y_pred, score_fn):
    thresholds = np.arange(0, 1+0.05, 0.05)
    scores = [score_fn(y, np.where(y_pred < threshold, 0, 1)) for threshold in thresholds]
    return np.max(scores)

st.header('Predictions')

preds = pd.DataFrame({'y':y_test, 'y_pred':y_pred})
n_y = len(y_test)

st.subheader('Statistics')
st.write('Max accuracy', round(get_max_score(y_test, y_pred, accuracy_score), 4))
st.write('Max $F_1$-score', round(get_max_score(y_test, y_pred, f1_score), 4))
st.write('Mean squared error', round(np.mean((y_test-y_pred)**2), 4))
st.write('Cross entropy loss', round(log_loss(y_test, y_pred), 4))

st.subheader('Histogram')
n_bins = st.slider('Number of bins', 2, n_y, int(np.sqrt(n_y)), step=1)

top_hist = alt.Chart(preds).mark_area(
    opacity=.4, interpolate='step'
).encode(
    alt.X('y_pred:Q', 
          bin=alt.Bin(maxbins=n_bins, extent=(0, 1)), 
          stack=None, 
          scale=alt.Scale(domain=(0, 1)),
         ),
    alt.Y('count(*):Q', 
          stack=None, 
         ),
    alt.Color('y:N'),
)

st.altair_chart(top_hist)

st.subheader('ROC curve')

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_df = pd.DataFrame({'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds})
roc_plot = alt.Chart(roc_df).mark_line(color = 'red', fillOpacity = 0.5
).encode(alt.X('fpr', title="False positive rate"), alt.Y('tpr', title="True positive rate")
).properties(height=400, width=400)

st.altair_chart(roc_plot)

st.header('Embeddings')

embed_data = pd.DataFrame({'x':X_embed[:, 0], 'y':X_embed[:, 1], 'label':y})

embed_plot = alt.Chart(embed_data).mark_point().encode(
    x='x:Q',
    y='y:Q',
    color=alt.Color('label:N'),
    tooltip=['x', 'y']
)
st.altair_chart(embed_plot, use_container_width=True)


df = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'b', 'c'])

c = alt.Chart(df).mark_circle().encode(x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
st.altair_chart(c, use_container_width=True)