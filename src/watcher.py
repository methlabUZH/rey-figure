import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from config import DIRECTORY_TO_WATCH
from restore_predict import predict_restored
import os


class Watcher:

    def __init__(self, models):
        self.observer = Observer()
        self.models = models

    def run(self):
        event_handler = Handler(self.models)

        if DIRECTORY_TO_WATCH is None:
            print("DIRECTORY_TO_WATCH must be provided to run demo. Please update config file...")
            exit(1)
        self.observer.schedule(event_handler, DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Error")

        self.observer.join()


class Handler(FileSystemEventHandler):

    def __init__(self, models):
        self.observer = Observer()
        self.models = models

    def on_any_event(self, event):
        if event.is_directory:
            return None

        elif event.event_type == 'created':
            # Take any action here when a file is first created.
            print("Received created event - %s." % event.src_path)
            path = event.src_path
            predict_restored(self.models, path, os.path.basename(path))


        elif event.event_type == 'modified':
            # Taken any action here when a file is modified.
            print("Received modified event - %s." % event.src_path)


if __name__ == '__main__':
    w = Watcher()
    w.run()
