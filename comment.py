from github import Auth, Github
import os

ACCESS_TOKEN =  os.environ["GH_ACCESSTOKEN"]

class GithubHandler:

    def __init__(self, repo_name: str = "deep-dive-mexico/aladdin-repo"):
        self.repo_name = repo_name

        # Authenticate
        self.authenticate()
        # get repo
        self.repo = self.g.get_repo(self.repo_name)
        # get prs
        self.prs = self.repo.get_pulls(state = "open", sort = "created", base = "master")
        # get prs numbers
        self.prs_nums = [pr.number for pr in self.prs]
        # create dict of prs
        self.prs_dict = {k: v for k, v in zip(self.prs_nums, self.prs)}

    def authenticate(self):
        # github authentication
        self.auth = Auth.Token(ACCESS_TOKEN)
        self.g = Github(auth=self.auth)

    def get_all_files(self):
        contents = self.repo.get_contents("")
        
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                contents.extend(self.repo.get_contents(file_content.path))
            
        self.all_files = contents

    def modify_pr(self, pr_num: int):
        pass

    def get_file_contents(self, filepath: str):
        if filepath in self.all_files:
            file_contents = self.repo.get_contents(filepath)
        else:
            file_contents = ""

        return file_contents
    
    def update_file_contents(self, filepath: str):
        contents = self.repo.get_contents(filepath, ref)


if __name__ == "__main__":
    print(os.environ["GH_ACCESSTOKEN"])
    #print(os.environ["pull_request_number"])
    GH = GithubHandler()
    
        
