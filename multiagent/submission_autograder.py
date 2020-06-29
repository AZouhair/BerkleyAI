from zipfile import ZipFile
import os, sys

project_name = "P2-2 - Multiagent"

submission_filename = "submission-p2-2.zip"

submitted_files = ['multiAgents.py']

if __name__ == '__main__':

    if os.path.exists(submission_filename):
        os.remove(submission_filename)
    
    with ZipFile(submission_filename, 'w') as zipObj:
        # Iterate over all the files in directory
        for filename in submitted_files:
            zipObj.write(filename)

    print("Now submit the file {} to project \'{}\' on Gradescope.".format(submission_filename, project_name))
