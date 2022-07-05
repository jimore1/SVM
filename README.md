# SVM

>     Support vector machine(SVM) is a supervised machine learning model that uses classification algorithms for <font color="blue">two-group classification probliems</font>.After giving an SVM model sets of labeled training data for each category,they're able to categorize new text.

#### 1.Two main advantages

* <font color="red">higher speed</font>
  
* <font color="red">better performance with a limited number of samples(in the thousands)</font>
  

>     Suitable for text classification problems,where it's common to have access to a dataset of at most a couple of thousands of tagged samples.

#### 2.Working process

###### 2.1.Linear data

>     The basics of SVM and how it works are best understood with a simple example.Let's imagine we have two tags:red and blue,and ourdata has two features:x and y.We want a classifier that,given a pair of (x,y) coordinates,outputs if it's either red or blue.We plot our already labeled training data on a plane:

![support vector machines svm](https://d33wubrfki0l68.cloudfront.net/bb6acfd742447bab1e4c619cad0d9aec9e6d58bd/7381d/static/52081a1b625e8ba22c00210d547b4f1a/ae702/plot_original.png)

>     A support vector machine takes these data points and outputs the hyperplane (which in two dimensions it’s simply a line) that best separates the tags. This line is the **decision boundary**: anything that falls to one side of it we will classify as _blue_, and anything that falls to the other as _red_.

![support vector machines svm](https://d33wubrfki0l68.cloudfront.net/ddde7b980851a9c12b72cee173debe8794962820/54f9b/static/57fd2448dfb67cfff990f32191463e80/ae702/plot_hyperplanes_2.png)

>     But, what exactly is _the best_ hyperplane? For SVM, it’s the one that maximizes the margins from both tags. In other words: the hyperplane (remember it's a line in this case) whose distance to the nearest element of each tag is the largest.

![support vector machines svm](https://d33wubrfki0l68.cloudfront.net/b33ca241f49a2ffc8b57a62558efd876410a49ac/56c8c/static/7002b9ebbacb0e878edbf30e8ff5b01c/ae702/plot_hyperplanes_annotated.png)

> You can check out [this video tutorial](https://www.youtube.com/watch?v=1NxnPkZM9bc) to learn exactly how this optimal hyperplane is found.

###### 2.2.Nonlinear data

> Now this example was easy, since clearly the data was linearly separable — we could draw a straight line to separate _red_ and _blue_. Sadly, usually things aren’t that simple. Take a look at this case:

![support vector machines svm](https://d33wubrfki0l68.cloudfront.net/a80bfcd16e1c9e1235f6afb112afba3988e31bb4/7276f/static/2631f704a0b3f6e31246294578a7d777/95f64/plot_circle_01.png)

>     It’s pretty clear that there’s not a linear decision boundary (a single straight line that separates both tags). However, the vectors are very clearly segregated and it looks as though it should be easy to separate them.

>     So here’s what we’ll do: we will add a third dimension. Up until now we had two dimensions: _x_ and _y_. We create a new _z_ dimension, and we rule that it be calculated a certain way that is convenient for us: _z = x² + y²_ (you’ll notice that’s the equation for a circle).

> This will give us a three-dimensional space. Taking a slice of that space, it looks like this:

![support vector machines svm](https://d33wubrfki0l68.cloudfront.net/1ecdc513142b40c1162efa613f6b4f8eb996cdbd/f9ac6/static/9f18ce83bab159464cc138a653e3fc63/dbc4f/plot_circle_02.png)

> What can SVM do with this? Let’s see:



>     That’s great! Note that since we are in three dimensions now, the hyperplane is a plane parallel to the _x_ axis at a certain _z_ (let’s say _z = 1_).

> What’s left is mapping it back to two dimensions:

![support vector machines svm](https://d33wubrfki0l68.cloudfront.net/6fc63b6e6ea9adc551d4245acfb6d1bc48d5ca69/2e42a/static/a4dc8a44f6a8adf55df920a602668a42/95f64/plot_circle_04.png)

>     And there we go! Our decision boundary is a circumference of radius 1, which separates both tags using SVM. Check out this 3d visualization to see another example of the same effect:

#### 3.The kernel trick

>     In our example we found a way to classify nonlinear data by cleverly mapping our space to a higher dimension. However, it turns out that calculating this transformation can get pretty computationally expensive: there can be a lot of new dimensions, each one of them possibly involving a complicated calculation. Doing this for every vector in the dataset can be a lot of work, so it’d be great if we could find a cheaper solution.

>     And we’re in luck! Here’s a trick: SVM doesn’t need the actual vectors to work its magic, it actually can get by only with the [dot products](https://en.wikipedia.org/wiki/Dot_product) between them. This means that we can sidestep the expensive calculations of the new dimensions.

> This is what we do instead:

* Imagine the new space we want:
  
  **_z = x² + y²_**
  
* Figure out what the dot product in that space looks like:
  
  **_a · b = xa · xb  +  ya · yb  +  za · zb_**
  
  **_a · b = xa · xb  +  ya · yb +  (xa² + ya²) · (xb² + yb²)_**
  
* Tell SVM to do its thing, but using the new dot product — we call this a [**kernel function**](https://www.quora.com/What-are-Kernels-in-Machine-Learning-and-SVM).
  

>     That’s it! That’s the **kernel trick**, which allows us to sidestep a lot of expensive calculations. Normally, the kernel is linear, and we get a linear classifier. However, by using a nonlinear kernel (like above) we can get a nonlinear classifier without transforming the data at all: we only change the dot product to that of the space that we want and SVM will happily chug along.

>     Note that the kernel trick isn’t actually part of SVM. It can be used with other linear classifiers such as logistic regression. A support vector machine only takes care of finding the decision boundary.

#### 4.Using SVM with Natural Language Classification

>     So, we can classify vectors in multidimensional space. Great! Now, we want to apply this algorithm for [text classification](https://monkeylearn.com/text-classification), and the first thing we need is a way to transform a piece of text into a vector of numbers so we can run SVM with them. In other words, which **features** do we have to use in order to classify texts using SVM?

>     The most common answer is word frequencies, [just like we did in Naive Bayes](https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/#feature-engineering). This means that we treat a text as a bag of words, and for every word that appears in that bag we have a feature. The value of that feature will be how frequent that word is in the text.

>     This method boils down to just counting how many times every word appears in a text and dividing it by the total number of words. So in the sentence _“All monkeys are primates but not all primates are monkeys”_ the word _monkeys_ has a frequency of 2/10 = 0.2, and the word _but_ has a frequency of 1/10 = 0.1 .

>     For a more advanced alternative for calculating frequencies, we can also use [TF-IDF](https://monkeylearn.com/blog/what-is-tf-idf/).

>     Now that we’ve done that, every text in our dataset is represented as a vector with thousands (or tens of thousands) of dimensions, every one representing the frequency of one of the words of the text. Perfect! This is what we feed to SVM for training. We can improve this by using [preprocessing techniques](https://monkeylearn.com/blog/text-cleaning/), like stemming, removing stopwords, and using n-grams.

#### 5.Choosing a kernel function

>     Now that we have the feature vectors, the only thing left to do is choosing a kernel function for our model. Every problem is different, and the kernel function depends on what the data looks like. In our example, our data was arranged in concentric circles, so we chose a kernel that matched those data points.

>     Taking that into account, what’s best for [natural language processing](https://monkeylearn.com/natural-language-processing/)? Do we need a nonlinear classifier? Or is the data linearly separable? It turns out that it’s best to stick to a linear kernel. Why?

>     Back in our example, we had two features. Some real uses of SVM in other fields may use tens or even hundreds of features. Meanwhile, NLP classifiers use _thousands_ of features, since they can have up to one for every word that appears in the training data. This changes the problem a little bit: while using nonlinear kernels may be a good idea in other cases, having this many features will end up making nonlinear kernels overfit the data. Therefore, it’s best to just stick to a good old linear kernel, which actually results in the best performance in these cases.

#### 6.Putting it all together

>     Now the only thing left to do is training! We have to take our set of labeled texts, convert them to vectors using word frequencies, and feed them to the algorithm — which will use our chosen kernel function — so it produces a model. Then, when we have a new unlabeled text that we want to classify, we convert it into a vector and give it to the model, which will output the tag of the text.
