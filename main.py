import subprocess
import sys
import os
import signal
import time
import threading

class ScriptManager:
    def __init__(self):
        self.current_process = None
        self.current_script = None
        self.scripts = {
            '1': 'pretest-mv.py',
            '2': 'sabi2.py',
            '3': 'cameratest.py',
            '4': 'test2.py',
            '5': 'drone_detection copy.py'
        }
        self.running = True

    def stop_current_script(self):
        """現在実行中のスクリプトを停止"""
        if self.current_process and self.current_process.poll() is None:
            print(f"\n{self.current_script} を停止中...")
            try:
                # より確実な停止処理
                if os.name == 'nt':  # Windows
                    # taskkillを使用してプロセスとその子プロセスを強制終了
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.current_process.pid)], 
                                 capture_output=True, timeout=5)
                else:  # Unix系
                    # プロセスグループ全体にSIGTERMを送信
                    try:
                        os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)
                    except:
                        # フォールバック: プロセスに直接SIGTERM
                        self.current_process.terminate()
                    time.sleep(2)
                    if self.current_process.poll() is None:
                        # プロセスグループ全体にSIGKILLを送信
                        try:
                            os.killpg(os.getpgid(self.current_process.pid), signal.SIGKILL)
                        except:
                            # フォールバック: プロセスに直接SIGKILL
                            self.current_process.kill()
                        time.sleep(1)
                
                # プロセスが確実に終了したか確認
                if self.current_process.poll() is None:
                    print(f"警告: {self.current_script} の停止に時間がかかっています")
                    time.sleep(2)
                    if self.current_process.poll() is None:
                        print(f"強制終了: {self.current_script}")
                        self.current_process.kill()
                
                print(f"{self.current_script} を停止しました")
                
            except subprocess.TimeoutExpired:
                print(f"タイムアウト: {self.current_script} の停止に失敗しました")
                try:
                    self.current_process.kill()
                except:
                    pass
            except Exception as e:
                print(f"停止中にエラーが発生しました: {e}")
                try:
                    self.current_process.kill()
                except:
                    pass
            finally:
                self.current_process = None
                self.current_script = None

    def start_script(self, script_name):
        """指定されたスクリプトを起動"""
        if not os.path.exists(script_name):
            print(f"エラー: {script_name} が見つかりません")
            return False
        
        try:
            print(f"\n{script_name} を起動中...")
            # プロセスグループを作成して、より確実に停止できるようにする
            if os.name == 'nt':  # Windows
                self.current_process = subprocess.Popen(
                    [sys.executable, script_name],
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:  # Unix系
                self.current_process = subprocess.Popen(
                    [sys.executable, script_name],
                    preexec_fn=os.setsid
                )
            self.current_script = script_name
            print(f"{script_name} を起動しました")
            return True
        except Exception as e:
            print(f"起動中にエラーが発生しました: {e}")
            return False

    def switch_script(self, key):
        """スクリプトの切り替え"""
        if key not in self.scripts:
            print(f"無効な選択です: {key}")
            return
        
        script_name = self.scripts[key]
        
        # 現在のスクリプトを停止
        self.stop_current_script()
        
        # 新しいスクリプトを起動
        if self.start_script(script_name):
            print(f"\n現在実行中: {script_name}")
        else:
            print(f"\n{script_name} の起動に失敗しました")

    def show_menu(self):
        """メニューを表示"""
        print("\n" + "="*50)
        print("スクリプト切り替えメニュー")
        print("="*50)
        for key, script in self.scripts.items():
            status = "実行中" if self.current_script == script else "停止中"
            print(f"{key}. {script} ({status})")
        print("q. 終了")
        print("="*50)

    def run(self):
        """メインループ"""
        print("スクリプト切り替えシステムを開始しました")
        print("数字キーでスクリプトを切り替え、'q'で終了します")
        
        while self.running:
            self.show_menu()
            
            try:
                choice = input("\n選択してください (1-5, q): ").strip().lower()
                
                if choice == 'q':
                    print("\n終了中...")
                    self.stop_current_script()
                    self.running = False
                    break
                elif choice in self.scripts:
                    self.switch_script(choice)
                else:
                    print("無効な選択です。1-5またはqを入力してください")
                
                time.sleep(1)  # 少し待機
                
            except KeyboardInterrupt:
                print("\n\nCtrl+Cが押されました。終了中...")
                self.stop_current_script()
                self.running = False
                break
            except Exception as e:
                print(f"エラーが発生しました: {e}")
                time.sleep(1)

def main():
    manager = ScriptManager()
    try:
        manager.run()
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")
    finally:
        print("スクリプト切り替えシステムを終了しました")

if __name__ == "__main__":
    main()
