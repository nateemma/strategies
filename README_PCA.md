# Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a statistical technique used to reduce the complexity of a dataset by finding the
underlying patterns and structures in the data.

In PCA, the data is transformed into a new coordinate system where the first axis, or principal component, captures the
most variation in the data. Subsequent axes capture progressively less variation, while being orthogonal (perpendicular)
to the previous axes. This process is repeated until all the variation in the data is accounted for or until the desired
number of components is reached.

The resulting principal components can then be used for data visualization, dimensionality reduction, or other analysis
tasks. PCA is commonly used in fields such as data science, machine learning, and finance, where large datasets with
many variables are encountered.

PCA can be used for compression by reducing the dimensionality of a dataset while retaining as much of its variance or
information as possible. By retaining only the most important principal components, which capture the majority of the
variability in the data, PCA can effectively compress the dataset.

For example, suppose you have a dataset with 100 variables, but you only need to retain the most important information
from the dataset. You can apply PCA to the dataset, which will result in a smaller set of principal components that
capture the most important patterns and structures in the data. By keeping only these principal components and
discarding the less important ones, you can effectively reduce the dimensionality of the data and compress it.

Note that the amount of compression achieved depends on the number of principal components retained. The fewer principal
components retained, the higher the compression ratio, but at the cost of losing some information. Therefore, the number
of principal components to retain should be chosen carefully to balance between compression and information loss.

## References

Towards Data Science - "A One-Stop Shop for Principal Component
Analysis": https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c

StatQuest - "Principal Component Analysis (PCA) clearly explained (2015)": https://www.youtube.com/watch?v=_UVHneBUBW0

Machine Learning Mastery - "A Gentle Introduction to Principal Component Analysis for Machine
Learning": https://machinelearningmastery.com/principal-components-analysis-for-dimensionality-reduction-in-python/

Principal Component Analysis - Wikipedia: https://en.wikipedia.org/wiki/Principal_component_analysis

Principal Component Analysis - Stanford University: https://web.stanford.edu/class/cs246/slides/05-dim_red.pdf