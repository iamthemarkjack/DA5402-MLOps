import os
from datetime import datetime
import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor

default_args = {
    "owner" : "Rohith Ramanan"
}

def read_in(file_path):
    with open(file_path, "r") as f:
        count = f.readline()
    return count

def send_email(ti, sender_credentials, receiver_email):
    count = ti.xcom_pull(task_ids="read_in")
    if count != 0:
        sender_email = sender_credentials["sender_email"]
        password = sender_credentials["password"]
        smtp_server = sender_credentials["smtp_server"]
        smtp_port = sender_credentials["smtp_port"]

        subject = "New Image-Caption Datas Added"
        body = f"<h1>The number of new image-caption datas that have been added to the database is : {count} </h1>"

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, msg.as_string())
                print("Email successfully sent!")
        except Exception as e:
            print(f"Error sending the email : {e}")


with DAG(
    default_args = default_args,
    dag_id = "DAG_2",
    description = "Senses for status file and sends email",
    start_date = datetime(2025, 2, 10),
    catchup = False,
    schedule_interval = "@hourly"
) as dag:

    task1 = FileSensor(
        task_id = "wait_for_file",
        filepath = "/opt/airflow/dags/run/status",
        fs_conn_id = "fs_default",
        poke_interval = 120,
        timeout = 360,
        mode = "poke"
    )

    task2 = PythonOperator(
        task_id = "read_in",
        python_callable = read_in,
        op_kwargs = {"file_path" : "/opt/airflow/dags/run/status"}
    )

    task3 = PythonOperator(
        task_id = "send_email",
        python_callable = send_email,
        op_kwargs = {"sender_credentials" : {"sender_email" : "Your email address",
                                             "password" : "Your email password or app password",
                                             "smtp_server" : "SMTP Server for your email",
                                             "smtp_port" : "SMTP Port for your email"},
                     "receiver_email" : "Receiver email address"}
    )

    task4 = BashOperator(
        task_id = "delete_file",
        bash_command = "rm -f /opt/airflow/dags/run/status"
    )

    task1 >> task2 >> task3 >> task4