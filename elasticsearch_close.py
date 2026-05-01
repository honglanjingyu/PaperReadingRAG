from elasticsearch import Elasticsearch
import gc


def force_close_es_connections():
    """强制关闭所有 ES 连接"""
    # 获取所有 Elasticsearch 实例并关闭
    for obj in gc.get_objects():
        if isinstance(obj, Elasticsearch):
            try:
                obj.close()
                print(f"关闭 ES 连接: {obj}")
            except:
                pass

    # 强制垃圾回收
    gc.collect()

    # 如果有 aiohttp 连接池，也需要关闭
    import aiohttp
    try:
        asyncio.get_event_loop().run_until_complete(aiohttp.ClientSession().close())
    except:
        pass


# 使用前调用
force_close_es_connections()