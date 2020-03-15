import requests
from bs4 import BeautifulSoup
import time


class GetQQZone():
    def __init__(self):
        self.BASE_URL = "https://user.qzone.qq.com/"

    def user_zone(self, user_id):
        user_zone_url = self.BASE_URL + user_id
        return user_zone_url

    def get_usel_photo(self, url):
        if url:
            try:
                headers = {
                    "cookie": "pgv_pvi=6640593920; RK=e24B3CrGV1; ptcz=16a4a2f3996f6c9565078dcd1a7bbf9ff4e90d6442b23db1601b9bae7b6a2e26; tvfe_boss_uuid=f5f496d3594a2568; pgv_pvid=8816190572; ptui_loginuin=1450136519; o_cookie=1450136519; pac_uid=1_1450136519; ied_qq=o1450136519; XWINDEXGREY=0; pgv_si=s5938448384; pgv_info=ssid=s7872693444; uin=o1450136519; skey=@d3vFTGbvC; p_uin=o1450136519; pt4_token=UNF0dsgMQAHzCJhu8lK5mBhv3H*F2gPr3Qw52JjAbwU_; p_skey=xA*8vCLDKFWKq7nSMnCcJUhSG8aX*YeHOdboZAbGWrA_; Loading=Yes; qz_screen=1440x900; 1450136519_todaycount=1; 1450136519_totalcount=19154; QZ_FE_WEBP_SUPPORT=1; cpu_performance_v8=2; __Q_w_s__QZN_TodoMsgCnt=1; rv2=80462841ACBF52B17E80B948BC11901BBB678E6FA4833ADB68; property20=10DED1B09E4B388197DC08621B5D1551649119AFECBF75509AEE2985F50BC5BEA165BAE0E78AAA23"
                }
                response = requests.get(url,headers=headers)
                if response.status_code == 200:
                    print(response.text)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    getQQZone = GetQQZone()
    userid = ""
    user_zone_url = getQQZone.user_zone(userid)
    getQQZone.get_usel_photo(user_zone_url)
