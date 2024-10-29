from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timedelta
import threading

i = 0

def job():
    global i
    i += 1
    print('Job executed', i)

def stop_job():
    threading.Thread(target=scheduler.shutdown).start()

scheduler = BlockingScheduler()
scheduler.add_job(job, 'cron', day_of_week='mon-fri', hour='07-18',second = '5,10,15,20,25,30,35,40,45,50,55', timezone='Europe/Paris', misfire_grace_time=15)
scheduler.add_job(stop_job, run_date=datetime.now() + timedelta(minutes=1))
scheduler.start()