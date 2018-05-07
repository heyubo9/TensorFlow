#coding = utf-8
import pymysql
import global_var

class mysql(object):
    """operate the mysql database"""

    def __init__(self):
        """the configuration is written in the main.conf
        """
        self._host = global_var.get_value('host')
        self._username = global_var.get_value('username')
        self._passwd = global_var.get_value('passwd')
        self._database = global_var.get_value('database')

    def connect(self):
        try:
            self._db = pymysql.connect(self._host, self._username, self._passwd, self._database)
            return 1
        except Exception as e:
            print(e)
            return 0

    def close(self):
        self._db.close()

    def find_label(self, quintuple):
        """find the flow label with the quintuple
        
        @param quintuple: the quintuple contains source/destination IP/port, protocol
            format: 'destination IP-source IP-dport-sport-protocol'
        @return string label: return the label string
        """
        sql = "select label from total where `Flow ID` = '%s'" % (quintuple)

        try:
            cursor = self._db.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            for row in result:
                label = row[0]
                break
            return label
        except Exception as e:
            return "NO RESULT"

    def find_type(self):
        """find the label in the dataset
        
        @return list label_list: return the label set in the dataset
        """
        sql = "select distinct label from total"
        type = []
        try:
            cursor = self._db.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            for row in result:
                if row[0] is not None:
                    type.append(row[0])
            return type
        except Exception as e:
            print(e)
            return -1

