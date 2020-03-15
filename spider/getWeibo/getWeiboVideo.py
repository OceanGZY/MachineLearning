import requests
import json
import time


class GetWeibo(object):
    def __init__(self):
        self.BASE_URL = "https://api.weibo.cn"

    def person_api_config(self, user_id):

        person_api = self.BASE_URL + "/2/profile?gsid=_2A25zOyExDeRxGedG71oT8i_KwzuIHXVuUTP5rDV6PUJbkdANLRDBkWpNUTjAvlFEHX3KFGGoh5-N8ipytf7LegMi&sensors_mark=0&wm=3333_2001&sensors_is_first_day=false&from=10A1293010&sensors_device_id=B9599546-6504-4CE0-8656-9CC88AB0ABF4&c=iphone&v_p=81&skin=default&v_f=1&s=39024cc1&b=0&networktype=wifi&lang=zh_CN&ua=iPhone9,1__weibo__10.1.2__iphone__os13.3&sflag=1&ft=11&aid=01AjFaDK_rA9S4xc_NM8weHEdoAnmQvFGya6ZnyyrQ-VV-VXA.&sourcetype=page&oriuicode=10000011_10000011&orifid=1005051848221687_-_new%24%24231093_-_recently&fid=1076036555451025&luicode=10000011&uicode=10000198&is_profile_lock=1&user_domain=" \
                     + str(user_id) + \
                     "&dynamic_follow_button_menu_enable=1&lfid=231093_-_recently&moduleID=pagecard&lcardid=2310930043___1848221687_6555451025&launchid=10000365--x"
        self.get_person_video_container(person_api)

    def get_person_video_container(self, url):
        time.sleep(3)
        try:
            response = requests.get(url)
            if response.status_code == 200:
                pjson = json.loads(response.text)
                for item in pjson["tabsInfo"]["tabs"]:
                    try:
                        tab_type = item["tab_type"]
                        tab_title = item["title"]
                        if tab_type == "video" and tab_title == "视频":
                            container_id = item["containerid"]
                            self.video_pre_api(container_id)
                    except Exception as e:
                        continue
        except Exception as e:
            print("get_person_video_container 错误", e)

    def video_pre_api(self, container_id):

        video_pre_api = self.BASE_URL + "/2/cardlist?gsid=_2A25zOyExDeRxGedG71oT8i_KwzuIHXVuUTP5rDV6PUJbkdANLRDBkWpNUTjAvlFEHX3KFGGoh5-N8ipytf7LegMi&sensors_mark=0&wm=3333_2001&sensors_is_first_day=false&from=10A1293010&sensors_device_id=B9599546-6504-4CE0-8656-9CC88AB0ABF4&c=iphone&v_p=81&skin=default&v_f=1&s=39024cc1&b=0&networktype=wifi&lang=zh_CN&ua=iPhone9,1__weibo__10.1.2__iphone__os13.3&sflag=1&ft=11&aid=01AjFaDK_rA9S4xc_NM8weHEdoAnmQvFGya6ZnyyrQ-VV-VXA.&oriuicode=10000011_10000011&page_interrupt_enable=1&moduleID=pagecard&orifid=1005051848221687_-_new%24%24231093_-_recently&count=20&luicode=10000011&containerid=" \
                        + str(container_id) + \
                        "&fid=2315676555451025&uicode=10000198&st_bottom_bar_new_style_enable=0&need_head_cards=0&feed_mypage_card_remould_enable=1&need_new_pop=1&page=1&client_key=75e2c9bcd65d13ac61c877ddaa458060&lfid=231093_-_recently&sourcetype=page&lcardid=2310930043___1848221687_6555451025&launchid=10000365--x"
        self.get_pre_video_url(video_pre_api, container_id)

    def get_pre_video_url(self, api, container_id):
        time.sleep(3)
        try:
            response = requests.get(api)
            if response.status_code == 200:
                vjson = json.loads(response.text)
                since_id = vjson["cardlistInfo"]["since_id"]
                for item in vjson["cards"][5:]:
                    try:
                        video_url = item["mblog"]["page_info"]["media_info"]["mp4_hd_url"]
                        video_name = item["mblog"]["page_info"]["content2"].strip() + "video"
                        self.get_video(video_url, video_name)
                    except Exception as e:
                        continue
                if since_id:
                    self.get_after_video_url(container_id, since_id)
            else:
                print(response.status_code)
        except Exception as e:
            print("get_pre_video_url 错误", e)

    def get_after_video_url(self, container_id, since_id):
        time.sleep(3)

        after_video_api = self.BASE_URL + "/2/cardlist?gsid=_2A25zOyExDeRxGedG71oT8i_KwzuIHXVuUTP5rDV6PUJbkdANLRDBkWpNUTjAvlFEHX3KFGGoh5-N8ipytf7LegMi&sensors_mark=0&wm=3333_2001&sensors_is_first_day=false&from=10A1293010&b=0&c=iphone&networktype=wifi&skin=default&v_p=81&v_f=1&s=39024cc1&sensors_device_id=B9599546-6504-4CE0-8656-9CC88AB0ABF4&lang=zh_CN&sflag=1&ua=iPhone9,1__weibo__10.1.2__iphone__os13.3&ft=11&aid=01AjFaDK_rA9S4xc_NM8weHEdoAnmQvFGya6ZnyyrQ-VV-VXA.&lfid=1005051848221687_-_new&since_id=" + str(
            since_id) + "&orifid=1005051848221687_-_new&count=20&luicode=10000011&containerid=" + str(
            container_id) + "&fid=2315676555451025&uicode=10000198&need_head_cards=0&need_new_pop=1&page=1&oriuicode=10000011&page_interrupt_enable=1&moduleID=pagecard&launchid=10000365--x"
        print("当前的", container_id, since_id)
        try:
            response = requests.get(after_video_api)
            if response.status_code == 200:
                vjson = json.loads(response.text)
                for item in vjson["cards"]:
                    try:
                        video_url = item["mblog"]["page_info"]["media_info"]["mp4_hd_url"]
                        video_name = item["mblog"]["page_info"]["content2"].strip() + "video"
                        if video_url:
                            self.get_video(video_url, video_name)
                    except Exception as e:
                        continue
                new_id = vjson["cardlistInfo"]["since_id"]
                print("新的下一页id", new_id)
                if new_id:
                    self.get_after_video_url(container_id, new_id)
        except Exception as e:
            print("get_after_video_url 错误", e)

    def get_video(self, url, name):
        time.sleep(1)
        try:
            response = requests.get(url)
            if response.status_code == 200:
                file = name + ".mp4"
                with open(file, "wb") as f:
                    f.write(response.content)
                    print(file, "保存成功")
        except Exception as e:
            print("get_video 错误", e)


if __name__ == "__main__":
    getweibo = GetWeibo()
    user_id = "5720049006"
    getweibo.person_api_config(user_id)
