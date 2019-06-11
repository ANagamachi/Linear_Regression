import numpy as np
#from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

a=2.0
b=-1.2
n=1000
x=np.random.rand(n,1)*10
#uniform random number x of [0, 10]
ep=np.random.randn(n,1)*0.1
y=a*x+b+ep

shf=np.random.permutation(n)
xy=np.concatenate(([x[shf],y[shf]]),axis=1)

val_rate=0.1
test_rate=0.1
tr_size=int(n*(1-val_rate-test_rate))
val_size=int(n*val_rate)
test_size=int(n*test_rate)

xy_tr=xy[:tr_size,:]
xy_val=xy[tr_size:tr_size+val_size,:]
xy_test=xy[tr_size+val_size:,:]
x_tr , y_tr= xy_tr[:,0].reshape(-1,1), xy_tr[:,1].reshape(-1,1)
x_val, y_val = xy_val[:,0].reshape(-1,1), xy_val[:,1].reshape(-1,1)
x_test, y_test = xy_test[:,0].reshape(-1,1), xy_test[:,1].reshape(-1,1)

fig=plt.figure()
plt.plot(x_test,y_test,'ro',label='test data')
plt.xlabel('x')
plt.ylabel('y')

def least_square():
    lm=linear_model.LinearRegression()
    lm.fit(x_tr,y_tr)
    print('回帰直線：y={0:.2f}x{1:.2f}'.format(lm.coef_[0,0],lm.intercept_[0]))

def gradient_descent():
    learning_rate = 0.01
    epochs = 1000
    theta=np.random.randn(2,1)
    x_trb=np.concatenate(([np.ones((tr_size,1)),x_tr]),axis=1)
    x_valb=np.concatenate(([np.ones((val_size,1)),x_val]),axis=1)
    msel=[]
    ims=[]
    for epoch in range(epochs):
        print('epoch---------------------------------------')
        print(epoch)
        mse_tr=1.0/tr_size*np.sum((x_trb.dot(theta) - y_tr)**2)
        mse_val=1.0/val_size*np.sum((x_valb.dot(theta) - y_val)**2)
        #learnfig.plot(epoch,mse_tr,'ro')
        #learnfig.plot(epoch,mse_tr,'bo')
        msel.append([epoch-1,mse_tr,mse_val])
        gradients = 2.0/tr_size * x_trb.T.dot(x_trb.dot(theta) - y_tr) #MSEの偏微分
        theta = theta - learning_rate * gradients
        print('gradients')
        print(gradients)
        print('theta')
        print(theta)
        x_testb=np.concatenate([np.ones((test_size,1)),x_test],axis=1)
        y_pred=x_testb.dot(theta)
        im=plt.plot(x_test,y_pred,color=cm.cool_r(float(epoch/epochs)),linewidth=1.0)
        ims.append(im)
        if (gradients[0,0] > 10**8 or mse_tr < 10**(-6)):
            break
    print('回帰直線：y={0:.3g}x{1:.3g}'.format(theta[1,0],theta[0,0]))
    mse_test=1.0/test_size*np.sum((y_pred - y_test)**2)
    print('test error:{:.3g}'.format(mse_test))
    ani = animation.ArtistAnimation(fig, ims, interval=400)
#    plt.savefig('pred.png')
#    ani.save("output.gif", writer="imagemagick")
    plt.show()
    plt.close()
    msear=np.array(msel)
    plt.plot(msear[:,0],msear[:,1],'ro',markersize=1,label='training')
    plt.plot(msear[:,0],msear[:,2],'bo',markersize=1,label='validation')
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.legend()
    plt.savefig('learn_curve.png')
    plt.close()

#least_square()
gradient_descent()
