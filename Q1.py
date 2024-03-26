import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.odr as odr


im = cv.imread ('Images/crop_field_cropped.jpg', cv.IMREAD_GRAYSCALE)
assert im is not None
im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

edges = cv.Canny(im, 550, 690)

indices = np.where(edges != [0])
x = indices[1]
y = -indices[0]


# Question 1

fig, ax = plt.subplots(1,2)
ax[0].imshow(im)
ax[0].set_title('Original')
ax[1].imshow(edges)
ax[1].set_title('Edges')
plt.show()

# Question 3 - Least-Squares using polyfit
m, c = np.polyfit(x, y, 1)  # Fit a linear model (degree = 1)
x_line = np.array([min(x), max(x)])
y_line = m * x_line + c

# Question 6 - Total Least Squares
def linear_model(params, x):
    m2, c2 = params
    y = m2 * x + c2
    return y
linear_odr = odr.Model(linear_model)
data = odr.RealData(x, y)
guess = [m, c]
odr_obj = odr.ODR(data, linear_odr, beta0=guess)

output = odr_obj.run()

m2 = output.beta[0]  
c2 = output.beta[1]   
sd_m = output.sd_beta[0]  
sd_c = output.sd_beta[1]  

x_line2 = np.array([min(x), max(x)])
y_line2 = m2 * x_line2 + c2

# Question 4 - Calculating Angle
angle = np.arctan(m) 
deg = np.rad2deg(angle) 
print("Estimated Crop Field Angle OLS:", deg)

# Question 7 - Angles
angle = np.arctan(m2) 
deg = np.rad2deg(angle) 
print("Estimated Crop Field Angle TLS:", deg)

# Question 2 - Plotting Edges
plt.scatter(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Extracted Edges")
plt.plot(x_line, y_line, color='red', label='Least Squares Line')
plt.plot(x_line2, y_line2, color='green', label='Total Least Squares Line')
plt.show()

#Question 11

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn import linear_model

def line_equation_from_points(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    magnitude = np.sqrt(delta_x*2 + delta_y*2)
    a = delta_x / magnitude
    b = -delta_y / magnitude
    d = (a * x1) + (b * y1)
    return a, b, d

img = cv.imread("Images\Crop_field_cropped.jpg", cv.IMREAD_GRAYSCALE)
assert img is not None, "Image not found"

edges = cv.Canny(img, 550, 690)

indicas = np.where(edges != 0)
x = indicas[1]
y = indicas[0]

X = x.reshape(-1, 1)
y = y.reshape(-1, 1)

ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y_ransac = ransac.predict(line_X)

plt.scatter(x[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers")
plt.scatter(x[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers")
plt.plot(line_X, line_y_ransac, color="cornflowerblue", linewidth=2, label="RANSAC Regressor")
plt.legend(loc="lower right")
plt.title("RANSAC Line Fitting")
plt.xticks([])
plt.yticks([])

plt.show()
