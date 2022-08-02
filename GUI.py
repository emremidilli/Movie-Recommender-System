import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from Movie_Recommender import PreProcessing, Recommendation

from threading import Thread


c_i_WINDOW_HEIGHT = 400
c_i_WINDOW_WIDTH = 525



def SetPositionOfWindow(wnd):
    
    wnd.resizable(0,0) #removes maximize button
    wnd.attributes('-topmost', True)

    iScreenWidth = wnd.winfo_screenwidth()
    iScreenHeigth = wnd.winfo_screenheight()
    iPositionRight = int((iScreenWidth/2) - (c_i_WINDOW_WIDTH/2))
    iPositionDown = int((iScreenHeigth/2) - (c_i_WINDOW_HEIGHT/2))
    
    
    wnd.geometry("{}x{}+{}+{}".format(c_i_WINDOW_WIDTH, c_i_WINDOW_HEIGHT, iPositionRight , iPositionDown ))
    
    

    
def ShowWelcomePage(thrtToWait):
    wndWelcome = tk.Tk()
    
    wndWelcome.overrideredirect(1)
    
    SetPositionOfWindow(wndWelcome)
    
    wndWelcome.grid_rowconfigure(0, weight=1)
    wndWelcome.grid_columnconfigure(0, weight=1)
    
    
    lblWelcome = tk.Label(text="Welcome to Movie Recommender System", font=('Helvetica 15 bold'))
    lblWelcome.grid(row=0,column=0)
    
    # While the thread is alive
    while thrtToWait.is_alive():
        # Update the root so it will keep responding
        wndWelcome.update()
   
    
    wndWelcome.destroy()
    
    


    
def ShowMainPage(dfPreprocessed):
    
    wndMain = tk.Tk()
    
    SetPositionOfWindow(wndMain)

    frameUI = tk.Frame(wndMain)
    
    lblUserId = tk.Label(frameUI , text="User ID:")
    txtUserId = tk.Entry(frameUI)
    
    
    
    frameUserHistory= tk.Frame(wndMain)
    lblUserHistory = tk.Label(frameUserHistory, text="User History:")

    tvUserHistory = ttk.Treeview(
        frameUserHistory,
        selectmode = 'browse'
        , columns=(1, 2 , 3), 
        show='headings',
        height=5
        )
    

    scbUserHistoryVer = ttk.Scrollbar(frameUserHistory, orient="vertical", command=tvUserHistory.yview) # command means update the yaxis view of the widget
    scbUserHistoryHor = ttk.Scrollbar(frameUserHistory, orient="horizontal", command=tvUserHistory.xview) # command means update the xaxis view of the widget


    tvUserHistory.configure( yscrollcommand =scbUserHistoryVer.set) # assign the scrollbars to the Treeview Widget 
    tvUserHistory.configure( xscrollcommand=scbUserHistoryHor.set)

    tvUserHistory.column("# 1",anchor='w', stretch='NO', width=200)
    tvUserHistory.column("# 2",anchor='w', stretch='NO', width=200)
    tvUserHistory.column("# 3",anchor='w', stretch='NO', width=100)
    
    tvUserHistory.heading(1, text='title')
    tvUserHistory.heading(2, text='genres')
    tvUserHistory.heading(3, text='rating')
    



    frameRecommendation= tk.Frame(wndMain)
    lblRecommendation = tk.Label(frameRecommendation, text="Recommendation:")

    tvRecommendation = ttk.Treeview(
        frameRecommendation,
        selectmode = 'browse'
        , columns=(1, 2), 
        show='headings',
        height=5
        )
    

    scbRecommendationVer = ttk.Scrollbar(frameRecommendation, orient="vertical", command=tvRecommendation.yview) # command means update the yaxis view of the widget
    scbRecommendationHor = ttk.Scrollbar(frameRecommendation, orient="horizontal", command=tvRecommendation.xview) # command means update the xaxis view of the widget


    tvRecommendation.configure( yscrollcommand =scbRecommendationVer.set) # assign the scrollbars to the Treeview Widget 
    tvRecommendation.configure( xscrollcommand=scbRecommendationHor.set)

    tvRecommendation.column("# 1",anchor='w', stretch='NO', width=250)
    tvRecommendation.column("# 2",anchor='w', stretch='NO', width=250)
    
    tvRecommendation.heading(1, text='title')
    tvRecommendation.heading(2, text='genres')



    
    
    def HandleButtonEvent():
        tvRecommendation.delete(*tvRecommendation.get_children())
        tvUserHistory.delete(*tvUserHistory.get_children())
            
        if txtUserId.get().isnumeric() == False:
            messagebox.showwarning(title=None, message='Please enter a positive integer number')
        else:
            iUserId = int(txtUserId.get())
            
            if dfPreprocessed[dfPreprocessed['user_id'] == iUserId].shape[0] == 0:
                messagebox.showwarning(title=None, message='This user id has never rated before')
            else:
            
                dfRecommendations, dfUserHistory = Recommendation.dfGetRecommendation(iUserId,dfPreprocessed)
                
                for ix, srs in dfUserHistory.iterrows():
                    tvUserHistory.insert('', tk.END, values=(srs['title'], srs['genres'], srs['rating'])) 
    
    
                for ix, srs in dfRecommendations.iterrows():
                    tvRecommendation.insert('', tk.END, values=(srs['title'], srs['genres']))
                    
        
    

    btnGetRecommendations = tk.Button(
        frameUI, 
        text="Get recommendations",
        command = HandleButtonEvent
        )
    
    
    frameUI.pack(side = 'top', fill = 'both', expand = True)
    frameUserHistory.pack(side = 'top', fill = 'both', expand = True)
    frameRecommendation.pack(side = 'top', fill = 'both', expand = True)
    
    lblUserId.pack(side='left')
    txtUserId.pack( side='left')
    btnGetRecommendations.pack(side='right')
    
    
    lblUserHistory.grid(row =0 , column = 0)
    tvUserHistory.grid(row =1, column = 0)
    scbUserHistoryVer.grid(row =1, column = 1,  sticky="nse")   
    scbUserHistoryHor.grid(row = 2, column = 0, sticky="swse") 
    
    
 
    lblRecommendation.grid(row =0 , column = 0)
    tvRecommendation.grid(row =1, column = 0)
    scbRecommendationVer.grid(row =1, column = 1,  sticky="nse")   
    scbRecommendationHor.grid(row = 2, column = 0, sticky="swse") 
    

    wndMain.mainloop()
 

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return 
    


thrtPreprocessing = ThreadWithReturnValue(target=PreProcessing.dfGetPreprocessedData)
thrtPreprocessing.start()

ShowWelcomePage(thrtPreprocessing)

dfPreprocessed = thrtPreprocessing.join()

ShowMainPage(dfPreprocessed)




